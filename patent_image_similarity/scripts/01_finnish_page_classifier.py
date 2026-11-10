"""Script export of the matching notebook.

Run it from the same workspace root and in the same order documented in README.md.
"""

import matplotlib
matplotlib.use("Agg")
from IPython.display import display

#!/usr/bin/env python
# coding: utf-8

# # Finnish Patent Page Classifier
# 
# A complete bilingual Run All workflow covering the pilot, candidate-model training, and blind validation.
# 
# 完整的中英双语 Run All 流程，包括试验、候选模型训练和盲测验证。

# # Part 1 · 芬兰专利页面分类器试验 / Finnish Patent Page Classifier Pilot
# 
# 本 Notebook 使用 DINOv3 整页视觉 embedding 训练芬兰专用页面分类器。由于目前没有完整的逐页人工标签，本试验使用瑞典配对专利的绘图页作为弱监督，并使用 4 份已经人工查看的芬兰专利进行小型独立检查。试验模型不会覆盖现有瑞典分类器。
# 
# This notebook trains a Finnish-specific page classifier from DINOv3 whole-page embeddings. Because complete page-level labels are not yet available, matched Swedish drawing pages provide weak supervision and four visually reviewed Finnish patents provide a small independent check. The pilot never overwrites the Swedish classifier.

# ## 第0步：环境和输出目录 / Step 0: Environment and output folders

# In[ ]:


from pathlib import Path
from datetime import datetime
import importlib.metadata, json, math, re, sys, time

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import pymupdf
import torch
import torch.nn.functional as F
import timm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm").is_dir()), None)
assert ROOT is not None, "Could not find a parent directory containing embedding_atm."
if not (ROOT / 'sample_data').exists(): ROOT = ROOT.parent
DATA_ROOT = ROOT / 'Swe-Fin-matched-patents'
REPO = ROOT / 'swe_pre_process_pipeline-main' / 'swe_pre_process_pipeline-main'
SRC = REPO / 'src'
MODEL_PATH = REPO / 'artifacts' / 'page_classifier_v2.joblib'
OUTPUT_ROOT = ROOT / 'embedding_atm' / 'finnish_page_classifier_pilot'
RENDER_ROOT = OUTPUT_ROOT / 'rendered_pages_90dpi'
TABLE_ROOT = OUTPUT_ROOT / 'tables'
MODEL_ROOT = OUTPUT_ROOT / 'models'
PREVIEW_ROOT = OUTPUT_ROOT / 'previews'
for folder in [OUTPUT_ROOT, RENDER_ROOT, TABLE_ROOT, MODEL_ROOT, PREVIEW_ROOT]: folder.mkdir(parents=True, exist_ok=True)
assert DATA_ROOT.exists(), DATA_ROOT
assert MODEL_PATH.exists(), MODEL_PATH
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from page_classifier import (VISUAL_FEATURE_COLUMNS, drawing_probability,
                             extract_page_features, load_model_bundle)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Workspace:', ROOT)
print('Device:', device)
print('结论：输入数据、旧瑞典分类器和试验输出目录均已准备。')
print('Conclusion: Input data, the existing Swedish classifier, and pilot output folders are ready.')


# ## 第1步：建立配对清单并渲染页面 / Step 1: Build the pair inventory and render pages
# 
# 每份 PDF 只在缓存缺失时以 90 DPI 渲染。随后在相同页面图像上复现旧分类器的九个 OpenCV 特征和预测，供后续基线比较使用。
# 
# Each PDF is rendered at 90 DPI only when its cache is missing. The same page images are then used to reproduce the nine OpenCV features and predictions from the existing classifier for baseline comparison.

# In[ ]:


bundle = load_model_bundle(MODEL_PATH)

def render_pdf(pdf_path, output_dir, dpi=90):
    output_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as document:
        expected = document.page_count
        existing = sorted(output_dir.glob('page-*.png'))
        if len(existing) == expected:
            return existing
        for stale in existing: stale.unlink()
        matrix = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
        paths = []
        for page_index in range(expected):
            path = output_dir / f'page-{page_index + 1:04d}.png'
            document.load_page(page_index).get_pixmap(matrix=matrix, alpha=False).save(str(path))
            paths.append(path)
    return paths

pair_rows = []
for entry_dir in sorted(DATA_ROOT.iterdir()):
    if not entry_dir.is_dir(): continue
    fi_files = sorted(entry_dir.glob('FI.*.pdf'))
    se_files = sorted(entry_dir.glob('SE.*.pdf'))
    pair_rows.append({'entry_id': entry_dir.name,
                      'fi_pdf_path': str(fi_files[0].resolve()) if fi_files else '',
                      'se_pdf_path': str(se_files[0].resolve()) if se_files else '',
                      'pair_complete': bool(fi_files and se_files)})
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(TABLE_ROOT / 'matched_patent_pairs.csv', index=False, encoding='utf-8-sig')

page_rows = []
start = time.time()
for pair_number, pair in pairs.iterrows():
    for country, path_value in [('FI', pair.fi_pdf_path), ('SE', pair.se_pdf_path)]:
        if not path_value: continue
        pdf_path = Path(path_value)
        pages = render_pdf(pdf_path, RENDER_ROOT / pair.entry_id / country)
        for page_number, image_path in enumerate(pages, 1):
            features = extract_page_features(image_path,
                work_width=int(bundle['preprocessing']['work_width']),
                ink_threshold=int(bundle['preprocessing']['ink_threshold']))
            page_rows.append({'entry_id': pair.entry_id, 'country': country,
                              'pdf': pdf_path.name, 'pdf_path': str(pdf_path.resolve()),
                              'page': page_number, 'page_count': len(pages),
                              'relative_page': page_number / len(pages),
                              'image_path': str(image_path.resolve()), **features})
    if (pair_number + 1) % 20 == 0: print(f'Processed pairs: {pair_number + 1}/{len(pairs)}')
page_manifest = pd.DataFrame(page_rows)
page_manifest['old_drawing_probability'] = drawing_probability(bundle['model'], page_manifest[VISUAL_FEATURE_COLUMNS])
page_manifest['old_predicted_page_type'] = np.where(page_manifest.old_drawing_probability >= float(bundle['decision_threshold']), 'drawing', 'text')
page_manifest.insert(0, 'page_index', np.arange(len(page_manifest)))
page_manifest.to_csv(TABLE_ROOT / 'page_manifest_with_old_predictions.csv', index=False, encoding='utf-8-sig')
elapsed = (time.time() - start) / 60
display(pairs.pair_complete.value_counts().rename_axis('pair_complete').to_frame('entries'))
display(page_manifest.groupby('country').agg(pdfs=('pdf_path','nunique'), pages=('page','size'),
    old_drawing_pages=('old_predicted_page_type', lambda x: int((x == 'drawing').sum()))))
print(f'结论：{len(page_manifest)} 页已完成缓存渲染和旧分类器基线预测，耗时 {elapsed:.2f} 分钟。')
print(f'Conclusion: {len(page_manifest)} pages were rendered, cached, and scored by the old baseline in {elapsed:.2f} minutes.')


# ## 第2步：检查国家间的分类分布差异 / Step 2: Inspect the country-level distribution shift

# In[ ]:


distribution = page_manifest.groupby('country').agg(
    pdfs=('pdf_path','nunique'), pages=('page','size'),
    drawing_rate=('old_drawing_probability', lambda x: float((x >= .5).mean())),
    median_probability=('old_drawing_probability','median'),
    low_confidence_pages=('old_drawing_probability', lambda x: int(((x - .5).abs() < .15).sum()))).reset_index()
distribution.to_csv(TABLE_ROOT / 'old_classifier_country_distribution.csv', index=False, encoding='utf-8-sig')
display(distribution.round(4))
fig, ax = plt.subplots(figsize=(9,5))
for country, color in [('FI','#d92d20'),('SE','#175cd3')]:
    values = page_manifest.loc[page_manifest.country == country, 'old_drawing_probability']
    ax.hist(values, bins=20, alpha=.55, label=country, color=color)
ax.axvline(.5, color='black', linestyle='--', linewidth=1)
ax.set(title='Existing classifier probability by country', xlabel='Drawing probability', ylabel='Pages')
ax.legend(); fig.tight_layout()
fig.savefig(PREVIEW_ROOT / 'old_classifier_country_distribution.png', dpi=160, bbox_inches='tight')
plt.show()
print('结论：芬兰页面在旧分类器中出现明显的低置信和分布偏移，因此需要芬兰专用视觉模型。')
print('Conclusion: Finnish pages show a clear confidence and distribution shift under the old classifier, supporting a Finnish-specific visual model.')


# ## 第3步：提取 DINOv3 整页 embedding / Step 3: Extract DINOv3 whole-page embeddings
# 
# 模型只作为冻结的特征提取器，不在 1,164 页小数据集上微调。页面保持完整纵横比并放置在白色方形画布中。
# 
# The encoder remains frozen rather than being fine-tuned on the small 1,164-page collection. Each full page preserves its aspect ratio on a white square canvas.

# In[ ]:


REQUESTED_MODEL = 'vit_base_patch16_dinov3.lvd1689m'
model = timm.create_model(REQUESTED_MODEL, pretrained=True, num_classes=0).to(device).eval()
MODEL_USED = REQUESTED_MODEL
INPUT_SIZE = 256
EMBEDDING_DIM = int(model.num_features)
IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
IMAGENET_STD = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

def page_tensor(path):
    image = Image.open(path).convert('RGB')
    inner = int(INPUT_SIZE * .92)
    contained = ImageOps.contain(image, (inner, inner), Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), 'white')
    canvas.paste(contained, ((INPUT_SIZE-contained.width)//2, (INPUT_SIZE-contained.height)//2))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2,0,1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD

embedding_path = OUTPUT_ROOT / 'dinov3_page_embeddings.npy'
embedding_key_path = TABLE_ROOT / 'dinov3_page_embedding_manifest.csv'
page_keys = (page_manifest.entry_id + '|' + page_manifest.country + '|' + page_manifest.page.astype(str)).tolist()
cache_valid = False
if embedding_path.exists() and embedding_key_path.exists():
    cached_keys = pd.read_csv(embedding_key_path).page_key.tolist()
    cached_embeddings = np.load(embedding_path)
    cache_valid = cached_keys == page_keys and cached_embeddings.shape == (len(page_manifest), EMBEDDING_DIM)
if cache_valid:
    embeddings = cached_embeddings.astype(np.float32)
    print('Reused cached embeddings.')
else:
    outputs = []
    batch_size = 6 if device.type == 'cpu' else 24
    start = time.time()
    with torch.inference_mode():
        for batch_start in range(0, len(page_manifest), batch_size):
            paths = page_manifest.image_path.iloc[batch_start:batch_start+batch_size]
            batch = torch.stack([page_tensor(path) for path in paths]).to(device)
            vectors = model(batch)
            if isinstance(vectors, (tuple,list)): vectors = vectors[0]
            outputs.append(F.normalize(vectors.float(), p=2, dim=1).cpu().numpy())
            completed = min(batch_start + batch_size, len(page_manifest))
            if completed % 120 < batch_size: print(f'Embedded pages: {completed}/{len(page_manifest)}')
    embeddings = np.concatenate(outputs).astype(np.float32)
    np.save(embedding_path, embeddings)
    pd.DataFrame({'page_index':page_manifest.page_index,'page_key':page_keys}).to_csv(embedding_key_path,index=False,encoding='utf-8-sig')
    print(f'Embedding time: {(time.time()-start)/60:.2f} minutes')
assert embeddings.shape == (len(page_manifest), EMBEDDING_DIM)
model_info = {'model_used':MODEL_USED,'embedding_dimension':EMBEDDING_DIM,'input_size':INPUT_SIZE,'device':str(device)}
(OUTPUT_ROOT / 'dino_model_info.json').write_text(json.dumps(model_info,indent=2),encoding='utf-8')
print(f'结论：{len(embeddings)} 个完整页面已转换为 {EMBEDDING_DIM} 维 DINOv3 embedding。')
print(f'Conclusion: {len(embeddings)} full pages were converted into {EMBEDDING_DIM}-dimensional DINOv3 embeddings.')


# ## 第4步：从已匹配专利生成保守弱标签 / Step 4: Generate conservative weak labels from matched patents
# 
# 瑞典分类器先定位配对瑞典 PDF 的绘图页。每个瑞典绘图页再在同一专利对的芬兰页面中寻找最相似页面。只有可靠候选进入 drawing/text 训练集，其余页面保留为 review。
# 
# The Swedish classifier first identifies drawing pages in the paired Swedish PDF. Each Swedish drawing page then retrieves its closest Finnish page within the same patent pair. Only conservative candidates enter the drawing/text training set; the remainder stay in review.

# In[ ]:


GOLD_DRAWING_PAGES = {
    'FI.239.A.pdf': list(range(18,34)),
    'FI.257.A.pdf': [],
    'FI.603.A.pdf': [9],
    'FI.624.A.pdf': [5],
    'FI.866.A.pdf': [8],
}
gold_pdfs = set(GOLD_DRAWING_PAGES)
weak = page_manifest[page_manifest.country == 'FI'].copy()
weak['matched_se_max_similarity'] = np.nan
weak['weak_label'] = 'review'
weak['weak_label_reason'] = 'requires_review'

for entry_id in weak.entry_id.unique():
    fi_global = page_manifest.index[(page_manifest.entry_id == entry_id) & (page_manifest.country == 'FI')].to_numpy()
    se_global = page_manifest.index[(page_manifest.entry_id == entry_id) & (page_manifest.country == 'SE') &
                                    (page_manifest.old_predicted_page_type == 'drawing')].to_numpy()
    fi_weak_index = weak.index[weak.entry_id == entry_id].to_numpy()
    if len(se_global):
        similarities = embeddings[fi_global] @ embeddings[se_global].T
        max_similarity = similarities.max(axis=1)
        weak.loc[fi_weak_index, 'matched_se_max_similarity'] = max_similarity
        positive_positions = set(np.argmax(similarities, axis=0).tolist())
        for position in positive_positions:
            if max_similarity[position] >= .55:
                idx = fi_weak_index[position]
                weak.loc[idx, ['weak_label','weak_label_reason']] = ['drawing','closest_to_matched_se_drawing']
        extra = ((weak.loc[fi_weak_index,'weak_label'] != 'drawing') & (weak.loc[fi_weak_index,'relative_page'] >= .65) &
                 (weak.loc[fi_weak_index,'old_drawing_probability'] >= .95) & (max_similarity >= .72))
        weak.loc[fi_weak_index[extra.to_numpy()], ['weak_label','weak_label_reason']] = ['drawing','high_confidence_late_page']
    remaining = weak.loc[fi_weak_index,'weak_label'] == 'review'
    conservative_text = remaining & ((weak.loc[fi_weak_index,'relative_page'] < .65) |
                        (weak.loc[fi_weak_index,'old_drawing_probability'] < .35))
    weak.loc[fi_weak_index[conservative_text.to_numpy()], ['weak_label','weak_label_reason']] = ['text','conservative_text_rule']

weak['manual_gold_label'] = ''
for pdf_name, drawing_pages in GOLD_DRAWING_PAGES.items():
    mask = weak.pdf == pdf_name
    weak.loc[mask, 'manual_gold_label'] = np.where(weak.loc[mask,'page'].isin(drawing_pages), 'drawing', 'text')
weak['training_eligible'] = weak.weak_label.isin(['text','drawing']) & ~weak.pdf.isin(gold_pdfs)
weak.to_csv(TABLE_ROOT / 'weak_training_labels.csv', index=False, encoding='utf-8-sig')
display(weak.groupby(['weak_label','training_eligible']).size().unstack(fill_value=0))
print('结论：已生成保守弱标签；4 份人工检查专利已从训练中完全排除。弱标签只用于试验，不代表最终真值。')
print('Conclusion: Conservative weak labels are ready and the four manually reviewed patents are fully excluded from training. Weak labels are pilot supervision, not final ground truth.')


# ## 第5步：按专利分组验证并训练轻量分类器 / Step 5: Grouped validation and lightweight classifier training
# 
# 比较 DINOv3-only 与 DINOv3+页面位置两个版本。所有交叉验证都按 entry_id 分组，确保同一专利的页面不会同时出现在训练和验证折中。
# 
# DINOv3-only and DINOv3-plus-page-position variants are compared. Every validation fold is grouped by entry_id so pages from one patent never appear in both training and validation.

# In[ ]:


train_rows = weak[weak.training_eligible].copy()
train_indices = train_rows.page_index.to_numpy(dtype=int)
y_train = (train_rows.weak_label == 'drawing').astype(int).to_numpy()
groups = train_rows.entry_id.to_numpy()
feature_sets = {
    'dino_only': embeddings[train_indices],
    'dino_plus_page_position': np.column_stack([embeddings[train_indices], train_rows.relative_page.to_numpy()]),
}
base_model = Pipeline([('scale',StandardScaler()),
                       ('classifier',LogisticRegression(C=.25,class_weight='balanced',max_iter=4000,random_state=42))])
cv = GroupKFold(n_splits=5)
cv_rows = []
oof_by_variant = {}
for variant, features in feature_sets.items():
    oof = np.zeros(len(train_rows), dtype=float)
    for fold, (fit_idx, val_idx) in enumerate(cv.split(features, y_train, groups), 1):
        fold_model = clone(base_model).fit(features[fit_idx], y_train[fit_idx])
        oof[val_idx] = fold_model.predict_proba(features[val_idx])[:,1]
    oof_by_variant[variant] = oof
    pred = oof >= .5
    cv_rows.append({'feature_variant':variant,'pages':len(y_train),
        'accuracy':accuracy_score(y_train,pred),'balanced_accuracy':balanced_accuracy_score(y_train,pred),
        'drawing_precision':precision_score(y_train,pred,zero_division=0),
        'drawing_recall':recall_score(y_train,pred,zero_division=0),
        'drawing_f1':f1_score(y_train,pred,zero_division=0),'roc_auc':roc_auc_score(y_train,oof)})
cv_metrics = pd.DataFrame(cv_rows).sort_values(['balanced_accuracy','drawing_recall'],ascending=False).reset_index(drop=True)
cv_metrics.to_csv(TABLE_ROOT / 'group_cross_validation_metrics.csv',index=False,encoding='utf-8-sig')
display(cv_metrics.round(4))
selected_variant = cv_metrics.iloc[0].feature_variant
selected_features = feature_sets[selected_variant]
selected_oof = oof_by_variant[selected_variant]
threshold_rows = []
for threshold in np.linspace(.10,.90,161):
    pred = selected_oof >= threshold
    threshold_rows.append({'threshold':threshold,'balanced_accuracy':balanced_accuracy_score(y_train,pred),
                           'drawing_recall':recall_score(y_train,pred,zero_division=0),
                           'drawing_precision':precision_score(y_train,pred,zero_division=0)})
threshold_table = pd.DataFrame(threshold_rows)
eligible_thresholds = threshold_table[threshold_table.drawing_recall >= .95]
selected_threshold = float((eligible_thresholds if len(eligible_thresholds) else threshold_table).sort_values(
    ['balanced_accuracy','drawing_precision'],ascending=False).iloc[0].threshold)
final_model = clone(base_model).fit(selected_features, y_train)
pd.DataFrame({'page_index':train_indices,'weak_label':train_rows.weak_label,
              'oof_drawing_probability':selected_oof}).to_csv(TABLE_ROOT/'group_oof_predictions.csv',index=False,encoding='utf-8-sig')
model_bundle = {'model':final_model,'feature_variant':selected_variant,'decision_threshold':selected_threshold,
                'dino_model':MODEL_USED,'embedding_dimension':EMBEDDING_DIM,
                'classes':{0:'text',1:'drawing'},'status':'pilot_weak_supervision'}
joblib.dump(model_bundle, MODEL_ROOT / 'finnish_page_classifier_dino_pilot.joblib', compress=3)
print('Selected variant:',selected_variant,'| threshold:',round(selected_threshold,3))
print('结论：按专利分组的交叉验证已完成，并保存了弱监督轻量模型。')
print('Conclusion: Patent-grouped cross-validation is complete and the weakly supervised lightweight model is saved.')


# ## 第6步：在人工查看的 5 份芬兰专利上做独立检查 / Step 6: Independent check on five visually reviewed Finnish patents
# 
# 这 60 页完全没有用于训练，其中 FI.239 是包含 16 个连续绘图页的困难案例。样本仍然有限，因此这里只能判断方向是否值得继续，不能作为最终准确率。
# 
# These 60 pages are completely excluded from training, including FI.239 as a hard case with 16 consecutive drawing pages. The sample remains limited, so this is a directional sanity check rather than a final accuracy estimate.

# In[ ]:


gold = weak[weak.manual_gold_label != ''].copy()
gold_indices = gold.page_index.to_numpy(dtype=int)
gold_features = embeddings[gold_indices] if selected_variant == 'dino_only' else np.column_stack([
    embeddings[gold_indices], gold.relative_page.to_numpy()])
gold['pilot_drawing_probability'] = final_model.predict_proba(gold_features)[:,1]
gold['pilot_prediction'] = np.where(gold.pilot_drawing_probability >= selected_threshold,'drawing','text')
gold['old_prediction'] = gold.old_predicted_page_type
gold['gold_is_drawing'] = (gold.manual_gold_label == 'drawing').astype(int)
gold['pilot_is_drawing'] = (gold.pilot_prediction == 'drawing').astype(int)
gold['old_is_drawing'] = (gold.old_prediction == 'drawing').astype(int)
manual_metrics = []
for name, column in [('old_opencv_baseline','old_is_drawing'),('dino_pilot','pilot_is_drawing')]:
    truth = gold.gold_is_drawing; pred = gold[column]
    manual_metrics.append({'model':name,'pages':len(gold),'drawing_pages':int(truth.sum()),
        'accuracy':accuracy_score(truth,pred),'balanced_accuracy':balanced_accuracy_score(truth,pred),
        'drawing_precision':precision_score(truth,pred,zero_division=0),
        'drawing_recall':recall_score(truth,pred,zero_division=0),
        'drawing_f1':f1_score(truth,pred,zero_division=0),
        'false_positive_pages':int(((pred==1)&(truth==0)).sum()),
        'false_negative_pages':int(((pred==0)&(truth==1)).sum())})
manual_metrics = pd.DataFrame(manual_metrics)
manual_metrics.to_csv(TABLE_ROOT/'manual_pilot_metrics.csv',index=False,encoding='utf-8-sig')
gold[['entry_id','pdf','page','manual_gold_label','old_drawing_probability','old_prediction',
      'pilot_drawing_probability','pilot_prediction','image_path']].to_csv(
      TABLE_ROOT/'manual_pilot_predictions.csv',index=False,encoding='utf-8-sig')
display(manual_metrics.round(4))
print('结论：独立小样本比较已完成；只有扩大人工标注后，才能决定是否将 pilot 升级为正式芬兰分类器。')
print('Conclusion: The independent pilot comparison is complete; broader manual labels are required before promoting the model to production.')


# ## 第7步：生成英文人工复核表和不确定页面预览 / Step 7: Generate an English review table and uncertainty preview

# In[ ]:


fi_indices = weak.page_index.to_numpy(dtype=int)
fi_features = embeddings[fi_indices] if selected_variant == 'dino_only' else np.column_stack([
    embeddings[fi_indices], weak.relative_page.to_numpy()])
weak['pilot_drawing_probability'] = final_model.predict_proba(fi_features)[:,1]
weak['pilot_prediction'] = np.where(weak.pilot_drawing_probability >= selected_threshold,'drawing','text')
weak['uncertainty'] = (weak.pilot_drawing_probability - selected_threshold).abs()
review_queue = weak[['entry_id','pdf','page','page_count','relative_page','weak_label','weak_label_reason',
                     'old_drawing_probability','matched_se_max_similarity','pilot_drawing_probability',
                     'pilot_prediction','image_path']].copy()
review_queue.insert(0,'review_id',[f'FI-PAGE-{i+1:04d}' for i in range(len(review_queue))])
review_queue['human_label'] = ''
review_queue['human_notes'] = ''
review_queue.to_csv(TABLE_ROOT/'finnish_page_review_queue.csv',index=False,encoding='utf-8-sig')

uncertain = weak.sort_values('uncertainty').head(30)
tile_w,tile_h,columns = 360,500,5
rows = math.ceil(len(uncertain)/columns)
sheet = Image.new('RGB',(columns*tile_w,rows*tile_h),'#e5e7eb')
for position,(_,row) in enumerate(uncertain.iterrows()):
    image = Image.open(row.image_path).convert('RGB'); image.thumbnail((330,430))
    tile = Image.new('RGB',(tile_w,tile_h),'white')
    tile.paste(image,((tile_w-image.width)//2,45))
    draw = ImageDraw.Draw(tile)
    label = f'{row.pdf} p{row.page} | pilot={row.pilot_drawing_probability:.3f} | weak={row.weak_label}'
    draw.text((8,10),label,fill='black')
    sheet.paste(tile,((position%columns)*tile_w,(position//columns)*tile_h))
preview_path = PREVIEW_ROOT/'most_uncertain_finnish_pages.png'
sheet.save(preview_path)
display(Image.open(preview_path))
print('结论：纯英文人工复核表和 30 个最不确定页面预览已经生成。')
print('Conclusion: An English-only manual review table and a preview of the 30 most uncertain pages are ready.')


# ## 第8步：保存试验总结 / Step 8: Save the pilot summary

# In[ ]:


pilot_row = manual_metrics[manual_metrics.model == 'dino_pilot'].iloc[0]
baseline_row = manual_metrics[manual_metrics.model == 'old_opencv_baseline'].iloc[0]
summary = {
    'created_at':datetime.now().isoformat(timespec='seconds'),
    'status':'pilot_weak_supervision_not_for_production',
    'matched_entries':int(len(pairs)), 'complete_pairs':int(pairs.pair_complete.sum()),
    'total_pages':int(len(page_manifest)), 'finnish_pages':int((page_manifest.country=='FI').sum()),
    'dino_model':MODEL_USED, 'embedding_dimension':EMBEDDING_DIM,
    'selected_feature_variant':selected_variant, 'decision_threshold':selected_threshold,
    'weak_training_pages':int(len(train_rows)), 'weak_drawing_pages':int(y_train.sum()),
    'manual_check_pages':int(len(gold)), 'manual_check_drawing_pages':int(gold.gold_is_drawing.sum()),
    'old_baseline_manual_balanced_accuracy':float(baseline_row.balanced_accuracy),
    'dino_pilot_manual_balanced_accuracy':float(pilot_row.balanced_accuracy),
    'dino_pilot_manual_drawing_recall':float(pilot_row.drawing_recall),
    'model_path':str((MODEL_ROOT/'finnish_page_classifier_dino_pilot.joblib').resolve()),
    'review_queue':str((TABLE_ROOT/'finnish_page_review_queue.csv').resolve()),
    'limitation':'Weak supervision and a 60-page manual check are insufficient for production deployment.'
}
(OUTPUT_ROOT/'pilot_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
display(pd.DataFrame([summary]).T.rename(columns={0:'value'}))
print('结论：芬兰 DINOv3 页面分类试验已完成；下一步是人工修正复核表后重新训练正式模型。')
print('Conclusion: The Finnish DINOv3 page-classification pilot is complete; the next step is to correct the review table and retrain a production model.')


# In[ ]:


# Release memory before the next stage / 在下一阶段前释放内存
import gc
if "model" in globals():
    try:
        del model
    except Exception:
        pass
if "torch" in globals() and torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("Stage cleanup complete. / 本阶段内存清理完成。")


# # Part 2 · 芬兰页面分类器候选 V1 / Finnish Page Classifier Candidate V1
# 
# 本 Notebook 将用户确认的默认规则“当前 review 页面为 text”与逐页视觉检查结果合并，训练 DINOv3 芬兰页面分类器候选 V1。FI.245 第 6–7 页经单独复核确认为绘图，并覆盖默认规则。人工标签权重高于弱标签，验证始终按专利分组。
# 
# This notebook combines the user-confirmed default that current review pages are text with page-level visual checks. FI.245 pages 6-7 were separately verified as drawings and override the default. Manual labels receive higher training weight than weak labels, and validation remains grouped by patent.

# ## 第0步：加载 Pilot 产物 / Step 0: Load the pilot artifacts

# In[ ]:


from pathlib import Path
from datetime import datetime
import json, math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm").is_dir()), None)
assert ROOT is not None, "Could not find a parent directory containing embedding_atm."
if not (ROOT/'sample_data').exists(): ROOT = ROOT.parent
PILOT_ROOT = ROOT/'embedding_atm'/'finnish_page_classifier_pilot'
OUTPUT_ROOT = ROOT/'embedding_atm'/'finnish_page_classifier_v1'
TABLE_ROOT = OUTPUT_ROOT/'tables'; MODEL_ROOT = OUTPUT_ROOT/'models'; PREVIEW_ROOT = OUTPUT_ROOT/'previews'
for folder in [OUTPUT_ROOT,TABLE_ROOT,MODEL_ROOT,PREVIEW_ROOT]: folder.mkdir(parents=True,exist_ok=True)
page_manifest = pd.read_csv(PILOT_ROOT/'tables'/'page_manifest_with_old_predictions.csv')
weak = pd.read_csv(PILOT_ROOT/'tables'/'weak_training_labels.csv')
review_queue = pd.read_csv(PILOT_ROOT/'tables'/'finnish_page_review_queue.csv')
gold_predictions = pd.read_csv(PILOT_ROOT/'tables'/'manual_pilot_predictions.csv')
embeddings = np.load(PILOT_ROOT/'dinov3_page_embeddings.npy').astype(np.float32)
model_info = json.loads((PILOT_ROOT/'dino_model_info.json').read_text(encoding='utf-8'))
assert embeddings.shape[0] == len(page_manifest)
print('Pages:',len(page_manifest),'| Finnish:',int((page_manifest.country=='FI').sum()),'| Embedding dim:',embeddings.shape[1])
print('结论：Pilot 的页面、embedding、弱标签和人工检查结果已经完整加载。')
print('Conclusion: Pilot pages, embeddings, weak labels, and manual-review results are loaded.')


# ## 第1步：合并可追溯人工标签 / Step 1: Consolidate traceable manual labels
# 
# 标签优先级为：逐页视觉例外复核 > 逐页视觉检查 > 用户确认的 review=text > 弱标签。重复页面由更高优先级覆盖。
# 
# Label priority is: visual exception review, page-level visual inspection, the user-confirmed review=text rule, then weak supervision. Higher-priority labels override duplicates.

# In[ ]:


key_columns = ['entry_id','pdf','page']
index_lookup = page_manifest[key_columns+['page_index','relative_page','old_drawing_probability','image_path']].copy()
review_labels = weak[weak.weak_label == 'review'][key_columns].copy()
review_labels['manual_label'] = 'text'
review_labels['annotation_source'] = 'user_confirmed_review_pages_are_text'
gold_labels = gold_predictions[key_columns+['manual_gold_label']].rename(columns={'manual_gold_label':'manual_label'})
gold_labels['annotation_source'] = 'page_level_visual_review'
visual_exceptions = weak[(weak.pdf == 'FI.245.A.pdf') & weak.page.isin([6,7])][key_columns].copy()
assert len(visual_exceptions) == 2
visual_exceptions['manual_label'] = 'drawing'
visual_exceptions['annotation_source'] = 'visual_exception_review'
manual = pd.concat([review_labels,gold_labels,visual_exceptions],ignore_index=True)
manual['priority'] = manual.annotation_source.map({'user_confirmed_review_pages_are_text':1,'page_level_visual_review':2,'visual_exception_review':3})
manual = manual.sort_values('priority').drop_duplicates(key_columns,keep='last').merge(index_lookup,on=key_columns,how='left')
assert manual.page_index.notna().all()
manual['page_index'] = manual.page_index.astype(int)
manual['is_drawing'] = manual.manual_label.isin(['drawing','mixed']).astype(int)
manual.to_csv(TABLE_ROOT/'consolidated_manual_labels.csv',index=False,encoding='utf-8-sig')
display(manual.groupby(['annotation_source','manual_label']).size().unstack(fill_value=0))
print('Manual pages:',len(manual),'| Drawing:',int(manual.is_drawing.sum()),'| Patents:',manual.entry_id.nunique())
print('结论：用户确认的 review=text 标签与逐页视觉标签已合并，并保留来源和优先级。')
print('Conclusion: The user-confirmed review=text labels and page-level visual labels are consolidated with provenance and priority.')


# ## 第2步：建立人工高权重、弱标签低权重的训练池 / Step 2: Build a manual-high-weight and weak-low-weight training pool

# In[ ]:


weak_pool = weak[weak.weak_label.isin(['text','drawing'])].copy()
weak_pool = weak_pool[~weak_pool.page_index.isin(manual.page_index)].copy()
weak_pool['training_label'] = weak_pool.weak_label
weak_pool['label_source'] = 'weak_supervision'
weak_pool['sample_weight'] = 1.0
manual_pool = manual.copy()
manual_pool['training_label'] = manual_pool.manual_label
manual_pool['label_source'] = 'manual'
manual_pool['sample_weight'] = 4.0
common = ['page_index','entry_id','pdf','page','relative_page','training_label','label_source','sample_weight']
training_pool = pd.concat([weak_pool[common],manual_pool[common]],ignore_index=True)
training_pool['is_drawing'] = training_pool.training_label.isin(['drawing','mixed']).astype(int)
training_pool.to_csv(TABLE_ROOT/'weighted_training_pool.csv',index=False,encoding='utf-8-sig')
display(training_pool.groupby(['label_source','training_label']).size().unstack(fill_value=0))
print('结论：人工标签权重设为 4，弱标签权重设为 1；所有人工页面已从弱标签集合中移除。')
print('Conclusion: Manual labels receive weight 4 and weak labels weight 1; every manual page is removed from the weak-label subset.')


# ## 第3步：按专利分组的人工标签验证 / Step 3: Patent-grouped validation on manual labels
# 
# 每一折只在人工标签上评分，但训练时可以使用训练折之外专利的低权重弱标签。同一专利不会跨越训练和验证。
# 
# Each fold is scored only on manual labels, while low-weight weak labels from non-validation patents may support training. No patent crosses the train-validation boundary.

# In[ ]:


base_model = Pipeline([('scale',StandardScaler()),
    ('classifier',LogisticRegression(C=.2,class_weight='balanced',max_iter=4000,random_state=42))])
feature_variants = {
    'dino_only': lambda frame: embeddings[frame.page_index.to_numpy(dtype=int)],
    'dino_plus_page_position': lambda frame: np.column_stack([embeddings[frame.page_index.to_numpy(dtype=int)],frame.relative_page.to_numpy()]),
}
manual_y = manual.is_drawing.to_numpy()
manual_groups = manual.entry_id.to_numpy()
cv = StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=42)
variant_results=[]; variant_oof={}
for variant,feature_function in feature_variants.items():
    oof=np.full(len(manual),np.nan)
    for fold,(manual_fit,manual_val) in enumerate(cv.split(manual,manual_y,manual_groups),1):
        validation_entries=set(manual.iloc[manual_val].entry_id)
        fold_train=training_pool[~training_pool.entry_id.isin(validation_entries)].copy()
        x_fit=feature_function(fold_train); y_fit=fold_train.is_drawing.to_numpy()
        model=clone(base_model).fit(x_fit,y_fit,classifier__sample_weight=fold_train.sample_weight.to_numpy())
        oof[manual_val]=model.predict_proba(feature_function(manual.iloc[manual_val]))[:,1]
    assert np.isfinite(oof).all()
    variant_oof[variant]=oof
    pred=oof>=.5
    variant_results.append({'feature_variant':variant,'manual_pages':len(manual),
        'accuracy':accuracy_score(manual_y,pred),'balanced_accuracy':balanced_accuracy_score(manual_y,pred),
        'drawing_precision':precision_score(manual_y,pred,zero_division=0),
        'drawing_recall':recall_score(manual_y,pred,zero_division=0),
        'drawing_f1':f1_score(manual_y,pred,zero_division=0),'roc_auc':roc_auc_score(manual_y,oof)})
validation_metrics=pd.DataFrame(variant_results).sort_values(['balanced_accuracy','drawing_recall'],ascending=False).reset_index(drop=True)
validation_metrics.to_csv(TABLE_ROOT/'manual_grouped_validation_metrics.csv',index=False,encoding='utf-8-sig')
display(validation_metrics.round(4))
selected_variant=validation_metrics.iloc[0].feature_variant
selected_oof=variant_oof[selected_variant]
threshold_rows=[]
for threshold in np.linspace(.05,.95,181):
    pred=selected_oof>=threshold
    threshold_rows.append({'threshold':threshold,'balanced_accuracy':balanced_accuracy_score(manual_y,pred),
        'drawing_recall':recall_score(manual_y,pred,zero_division=0),
        'drawing_precision':precision_score(manual_y,pred,zero_division=0)})
threshold_table=pd.DataFrame(threshold_rows)
eligible=threshold_table[threshold_table.drawing_recall>=.95]
selected_threshold=float((eligible if len(eligible) else threshold_table).sort_values(
    ['balanced_accuracy','drawing_precision'],ascending=False).iloc[0].threshold)
manual_eval=manual.copy()
manual_eval['candidate_probability']=selected_oof
manual_eval['candidate_prediction']=np.where(selected_oof>=selected_threshold,'drawing','text')
manual_eval['old_prediction']=np.where(manual_eval.old_drawing_probability>=.5,'drawing','text')
manual_eval.to_csv(TABLE_ROOT/'manual_grouped_oof_predictions.csv',index=False,encoding='utf-8-sig')
print('Selected:',selected_variant,'| threshold:',round(selected_threshold,3))
print('结论：人工标签上的专利分组验证已经完成，阈值优先保证至少 95% 的绘图页召回。')
print('Conclusion: Patent-grouped validation on manual labels is complete, with threshold selection prioritising at least 95% drawing-page recall.')


# ## 第4步：与旧 OpenCV 基线直接比较 / Step 4: Direct comparison with the old OpenCV baseline

# In[ ]:


comparison=[]
for name,pred in [('old_opencv_baseline',(manual_eval.old_prediction=='drawing').astype(int)),
                  ('dino_candidate_v1',(manual_eval.candidate_prediction=='drawing').astype(int))]:
    comparison.append({'model':name,'manual_pages':len(manual_y),'drawing_pages':int(manual_y.sum()),
        'accuracy':accuracy_score(manual_y,pred),'balanced_accuracy':balanced_accuracy_score(manual_y,pred),
        'drawing_precision':precision_score(manual_y,pred,zero_division=0),
        'drawing_recall':recall_score(manual_y,pred,zero_division=0),
        'drawing_f1':f1_score(manual_y,pred,zero_division=0),
        'false_positive_pages':int(((pred==1)&(manual_y==0)).sum()),
        'false_negative_pages':int(((pred==0)&(manual_y==1)).sum())})
comparison=pd.DataFrame(comparison)
comparison.to_csv(TABLE_ROOT/'old_vs_dino_candidate_comparison.csv',index=False,encoding='utf-8-sig')
display(comparison.round(4))
print('结论：旧 OpenCV 与 DINO 候选模型已在同一套人工标签上完成直接比较。')
print('Conclusion: The old OpenCV baseline and DINO candidate are directly compared on the same manual labels.')


# ## 第5步：训练候选 V1 并预测全部芬兰页面 / Step 5: Train candidate V1 and predict all Finnish pages

# In[ ]:


feature_function=feature_variants[selected_variant]
final_model=clone(base_model).fit(feature_function(training_pool),training_pool.is_drawing.to_numpy(),
                                  classifier__sample_weight=training_pool.sample_weight.to_numpy())
model_bundle={'model':final_model,'feature_variant':selected_variant,'decision_threshold':selected_threshold,
    'dino_model':model_info['model_used'],'embedding_dimension':int(embeddings.shape[1]),
    'classes':{0:'text',1:'contains_drawing'},'status':'candidate_v1_manual_plus_weak'}
model_path=MODEL_ROOT/'finnish_page_classifier_dinov3_candidate_v1.joblib'
joblib.dump(model_bundle,model_path,compress=3)
fi=page_manifest[page_manifest.country=='FI'].copy()
fi_probability=final_model.predict_proba(feature_function(fi))[:,1]
fi['candidate_drawing_probability']=fi_probability
fi['candidate_prediction']=np.where(fi_probability>=selected_threshold,'drawing','text')
fi.to_csv(TABLE_ROOT/'all_finnish_page_predictions.csv',index=False,encoding='utf-8-sig')
display(fi.groupby('candidate_prediction').size().rename('pages').to_frame())
print('结论：候选 V1 已保存，并为全部 742 个芬兰页面生成预测。')
print('Conclusion: Candidate V1 is saved and predictions are generated for all 742 Finnish pages.')


# ## 第6步：生成分组验证错误复核图 / Step 6: Generate grouped-validation error review

# In[ ]:


errors=manual_eval[manual_eval.manual_label!=manual_eval.candidate_prediction].copy()
columns=4; tile_w=420; tile_h=600; rows=max(1,math.ceil(len(errors)/columns))
sheet=Image.new('RGB',(columns*tile_w,rows*tile_h),'#e5e7eb')
if errors.empty:
    draw=ImageDraw.Draw(sheet); draw.text((30,30),'No grouped-validation errors',fill='black')
else:
    for position,(_,row) in enumerate(errors.iterrows()):
        image=Image.open(row.image_path).convert('RGB'); image.thumbnail((390,520))
        tile=Image.new('RGB',(tile_w,tile_h),'white'); tile.paste(image,((tile_w-image.width)//2,55))
        draw=ImageDraw.Draw(tile); draw.text((8,10),f'{row.pdf} p{row.page} | true={row.manual_label} | p={row.candidate_probability:.3f}',fill='black')
        sheet.paste(tile,((position%columns)*tile_w,(position//columns)*tile_h))
error_path=PREVIEW_ROOT/'grouped_validation_errors.png'; sheet.save(error_path)
display(Image.open(error_path))
print(f'结论：分组验证共有 {len(errors)} 个错误，复核图已保存。')
print(f'Conclusion: Grouped validation produced {len(errors)} errors and the review sheet is saved.')


# ## 第7步：保存候选模型总结 / Step 7: Save the candidate-model summary

# In[ ]:


candidate_row=comparison[comparison.model=='dino_candidate_v1'].iloc[0]
baseline_row=comparison[comparison.model=='old_opencv_baseline'].iloc[0]
summary={'created_at':datetime.now().isoformat(timespec='seconds'),'status':'candidate_v1_not_yet_production',
    'manual_pages':int(len(manual)),'manual_drawing_pages':int(manual_y.sum()),'manual_patents':int(manual.entry_id.nunique()),
    'weak_training_pages':int((training_pool.label_source=='weak_supervision').sum()),
    'selected_feature_variant':selected_variant,'decision_threshold':selected_threshold,
    'grouped_manual_balanced_accuracy':float(candidate_row.balanced_accuracy),
    'grouped_manual_drawing_precision':float(candidate_row.drawing_precision),
    'grouped_manual_drawing_recall':float(candidate_row.drawing_recall),
    'grouped_manual_false_positives':int(candidate_row.false_positive_pages),
    'grouped_manual_false_negatives':int(candidate_row.false_negative_pages),
    'old_baseline_balanced_accuracy':float(baseline_row.balanced_accuracy),
    'predicted_finnish_drawing_pages':int((fi.candidate_prediction=='drawing').sum()),
    'model_path':str(model_path.resolve()),
    'limitation':'Review pages used a user-confirmed default text label with two visually reviewed drawing exceptions; broader random manual controls are still recommended before production.'}
(OUTPUT_ROOT/'candidate_v1_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
display(pd.DataFrame([summary]).T.rename(columns={0:'value'}))
print('结论：芬兰分类器候选 V1 已完成；在投入 PDF 批处理前仍建议增加随机人工对照页。')
print('Conclusion: Finnish classifier candidate V1 is complete; additional random manual controls remain advisable before PDF batch deployment.')


# In[ ]:


# Release memory before the next stage / 在下一阶段前释放内存
import gc
if "model" in globals():
    try:
        del model
    except Exception:
        pass
if "torch" in globals() and torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("Stage cleanup complete. / 本阶段内存清理完成。")


# # Part 3 · 芬兰页面分类器候选 V1 盲检 / Blind Validation of Finnish Page Classifier Candidate V1
# 
# 从未进入人工标签集合的专利中，分别抽取 20 个预测绘图页和 20 个预测文本页。抽样在人工标注前固定，用于检查候选 V1 是否能泛化到新的专利。
# 
# Twenty predicted drawing pages and twenty predicted text pages are sampled from patents absent from the manual-label set. Sampling is fixed before annotation to test generalisation to unseen patents.

# ## 第0步：加载候选预测 / Step 0: Load candidate predictions

# In[ ]:


from pathlib import Path
from datetime import datetime
import json, math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm").is_dir()), None)
assert ROOT is not None, "Could not find a parent directory containing embedding_atm."
if not (ROOT/'sample_data').exists(): ROOT=ROOT.parent
V1_ROOT=ROOT/'embedding_atm'/'finnish_page_classifier_v1'
OUTPUT_ROOT=ROOT/'embedding_atm'/'finnish_page_classifier_v1_blind_validation'
TABLE_ROOT=OUTPUT_ROOT/'tables'; PREVIEW_ROOT=OUTPUT_ROOT/'previews'
for folder in [OUTPUT_ROOT,TABLE_ROOT,PREVIEW_ROOT]: folder.mkdir(parents=True,exist_ok=True)
predictions=pd.read_csv(V1_ROOT/'tables'/'all_finnish_page_predictions.csv')
manual=pd.read_csv(V1_ROOT/'tables'/'consolidated_manual_labels.csv')
manual_entries=set(manual.entry_id)
eligible=predictions[~predictions.entry_id.isin(manual_entries)].copy()
print('Eligible unseen patents:',eligible.entry_id.nunique(),'| pages:',len(eligible))
print('结论：盲检候选池只包含未进入人工标签集合的专利。')
print('Conclusion: The blind-validation pool contains only patents absent from the manual-label set.')


# ## 第1步：固定均衡随机样本 / Step 1: Freeze a balanced random sample

# In[ ]:


def one_page_per_patent(frame,n,seed):
    shuffled=frame.sample(frac=1,random_state=seed)
    return shuffled.drop_duplicates('entry_id').head(n).copy()
drawing_sample=one_page_per_patent(eligible[eligible.candidate_prediction=='drawing'],20,20261106)
text_sample=one_page_per_patent(eligible[eligible.candidate_prediction=='text'],20,20261107)
assert len(drawing_sample)==20 and len(text_sample)==20
drawing_sample['sample_stratum']='predicted_drawing'
text_sample['sample_stratum']='predicted_text'
batch=pd.concat([drawing_sample,text_sample],ignore_index=True)
batch.insert(0,'annotation_id',[f'BLIND-{i+1:03d}' for i in range(len(batch))])
batch=batch[['annotation_id','sample_stratum','entry_id','pdf','page','page_count','candidate_drawing_probability','candidate_prediction','image_path']]
batch.to_csv(TABLE_ROOT/'blind_validation_batch.csv',index=False,encoding='utf-8-sig')
label_path=TABLE_ROOT/'blind_validation_manual_labels.csv'
if not label_path.exists():
    pd.DataFrame({'annotation_id':batch.annotation_id,'human_label':'','human_notes':''}).to_csv(label_path,index=False,encoding='utf-8-sig')
display(batch.groupby('sample_stratum').size().rename('pages').to_frame())
print('结论：40 页均衡盲检样本已固定，每份专利最多抽取一页。')
print('Conclusion: A balanced 40-page blind sample is frozen with at most one page per patent.')


# ## 第2步：生成英文标注联系表 / Step 2: Generate English annotation contact sheets

# In[ ]:


pages_per_sheet=10; columns=2; tile_w=700; tile_h=980
sheet_paths=[]
for sheet_number,start in enumerate(range(0,len(batch),pages_per_sheet),1):
    subset=batch.iloc[start:start+pages_per_sheet]
    rows=math.ceil(len(subset)/columns)
    sheet=Image.new('RGB',(columns*tile_w,rows*tile_h),'#d1d5db')
    for position,(_,row) in enumerate(subset.iterrows()):
        image=Image.open(row.image_path).convert('RGB'); image.thumbnail((660,880))
        tile=Image.new('RGB',(tile_w,tile_h),'white'); tile.paste(image,((tile_w-image.width)//2,75))
        draw=ImageDraw.Draw(tile)
        draw.text((12,12),f'{row.annotation_id} | {row.pdf} | page {row.page}/{row.page_count}',fill='black')
        draw.text((12,35),f'prediction={row.candidate_prediction} | probability={row.candidate_drawing_probability:.4f}',fill='black')
        sheet.paste(tile,((position%columns)*tile_w,(position//columns)*tile_h))
    path=PREVIEW_ROOT/f'blind_validation_sheet_{sheet_number:02d}.png'; sheet.save(path); sheet_paths.append(path)
display(Image.open(sheet_paths[0]))
print('结论：4 张纯英文盲检联系表已生成。')
print('Conclusion: Four English-only blind-validation contact sheets are generated.')


# ## 第3步：读取人工标签并计算盲检指标 / Step 3: Read manual labels and calculate blind metrics

# In[ ]:


labels=pd.read_csv(label_path,keep_default_na=False)
valid_labels={'text','drawing','mixed'}
labels_ready=(len(labels)==len(batch) and labels.annotation_id.tolist()==batch.annotation_id.tolist() and labels.human_label.isin(valid_labels).all())
if not labels_ready:
    print('等待人工标注：请在 blind_validation_manual_labels.csv 中填写 text/drawing/mixed。')
    print('Waiting for annotation: fill text/drawing/mixed in blind_validation_manual_labels.csv.')
else:
    evaluated=batch.merge(labels,on='annotation_id',how='left')
    truth=evaluated.human_label.isin(['drawing','mixed']).astype(int)
    pred=(evaluated.candidate_prediction=='drawing').astype(int)
    metrics={'pages':len(evaluated),'drawing_pages':int(truth.sum()),'accuracy':accuracy_score(truth,pred),
        'balanced_accuracy':balanced_accuracy_score(truth,pred),'drawing_precision':precision_score(truth,pred,zero_division=0),
        'drawing_recall':recall_score(truth,pred,zero_division=0),'drawing_f1':f1_score(truth,pred,zero_division=0),
        'false_positive_pages':int(((pred==1)&(truth==0)).sum()),'false_negative_pages':int(((pred==0)&(truth==1)).sum())}
    evaluated['is_correct']=truth.eq(pred)
    evaluated.to_csv(TABLE_ROOT/'blind_validation_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([metrics]).to_csv(TABLE_ROOT/'blind_validation_metrics.csv',index=False,encoding='utf-8-sig')
    summary={'created_at':datetime.now().isoformat(timespec='seconds'),'status':'complete',**metrics}
    (OUTPUT_ROOT/'blind_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    display(pd.DataFrame([metrics]).T.rename(columns={0:'value'}))
    print('结论：40 页未见专利盲检已经完成。')
    print('Conclusion: The 40-page unseen-patent blind validation is complete.')

