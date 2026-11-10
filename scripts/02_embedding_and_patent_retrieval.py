"""Script export of the matching notebook.

Run it from the same workspace root and in the same order documented in README.md.
"""

import matplotlib
matplotlib.use("Agg")
from IPython.display import display

#!/usr/bin/env python
# coding: utf-8

# # DINOv3 Embedding and Swedish–Finnish Patent Retrieval
# 
# A complete bilingual Run All workflow from drawing-page segmentation to Drawing-page fallback. SIFT/RANSAC computation is outside this release.
# 
# 完整的中英双语 Run All 流程，从绘图页分割一直到 Drawing-page fallback；本版本不运行 SIFT/RANSAC。

# # Part 1 · 芬兰绘图页 YOLO 分割 / YOLO Segmentation of Finnish Drawing Pages
# 
# 本 Notebook 使用芬兰页面分类器候选 V1 选择的绘图页，以 300 DPI 从原始 PDF 重新渲染，并使用瑞典预处理仓库的 `best.pt` 与相同连通域遮罩逻辑提取单独图形。每份芬兰专利拥有独立目录，并为后续 embedding 预留文件夹。
# 
# This notebook re-renders Finnish pages selected by classifier candidate V1 at 300 DPI and extracts individual figures with the Swedish preprocessing repository's `best.pt` and connected-component masking logic. Every Finnish patent receives an independent folder with a reserved embedding directory.

# ## 第0步：环境和路径 / Step 0: Environment and paths

# In[ ]:


from pathlib import Path
from datetime import datetime
import hashlib, importlib.metadata, json, math, time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image
import torch
from ultralytics import YOLO

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm").is_dir()), None)
assert ROOT is not None, "Could not find a parent directory containing embedding_atm."
if not (ROOT/'sample_data').exists(): ROOT=ROOT.parent
REPO=ROOT/'swe_pre_process_pipeline-main'/'swe_pre_process_pipeline-main'
MODEL_PATH=REPO/'artifacts'/'best.pt'
PREDICTION_PATH=ROOT/'embedding_atm'/'finnish_page_classifier_v1'/'tables'/'all_finnish_page_predictions.csv'
OUTPUT_ROOT=ROOT/'embedding_atm'/'finland_segmentation'
PATENT_ROOT=OUTPUT_ROOT/'patents'; TABLE_ROOT=OUTPUT_ROOT/'tables'; PREVIEW_ROOT=OUTPUT_ROOT/'previews'; LOG_ROOT=OUTPUT_ROOT/'logs'
for folder in [OUTPUT_ROOT,PATENT_ROOT,TABLE_ROOT,PREVIEW_ROOT,LOG_ROOT]: folder.mkdir(parents=True,exist_ok=True)
assert MODEL_PATH.exists(),MODEL_PATH
assert PREDICTION_PATH.exists(),PREDICTION_PATH
device=0 if torch.cuda.is_available() else 'cpu'
print('Torch:',torch.__version__,'| Ultralytics:',importlib.metadata.version('ultralytics'),'| Device:',device)
print('结论：芬兰页面预测、仓库 YOLO 权重和输出目录已准备。')
print('Conclusion: Finnish page predictions, repository YOLO weights, and output folders are ready.')


# ## 第1步：固定芬兰绘图页清单 / Step 1: Freeze the Finnish drawing-page manifest

# In[ ]:


all_predictions=pd.read_csv(PREDICTION_PATH)
selected=all_predictions[all_predictions.candidate_prediction=='drawing'].copy()
selected=selected.sort_values(['entry_id','pdf','page']).reset_index(drop=True)
selected.insert(0,'segmentation_page_id',[f'FI-DRAW-{i+1:04d}' for i in range(len(selected))])
assert len(selected)==151, f'Expected 151 drawing pages, found {len(selected)}'
assert selected.pdf_path.map(lambda p:Path(p).exists()).all()
selected.to_csv(TABLE_ROOT/'selected_finnish_drawing_pages.csv',index=False,encoding='utf-8-sig')
display(selected.groupby('pdf').size().describe().to_frame('drawing_pages_per_pdf'))
print('Drawing pages:',len(selected),'| Finnish PDFs:',selected.pdf.nunique(),'| Matched entries:',selected.entry_id.nunique())
print('结论：151 个芬兰绘图页清单已固定，后续 Run All 使用同一输入。')
print('Conclusion: The 151-page Finnish drawing manifest is frozen for reproducible Run All execution.')


# ## 第2步：以 300 DPI 重绘并建立每专利目录 / Step 2: Re-render at 300 DPI and create per-patent folders
# 
# 分类使用的 90 DPI 页面只适合页面判断。分割阶段从原始 PDF 重绘，以保留线条细节。已存在且尺寸有效的页面会被复用。
# 
# The 90-DPI classifier images are suitable only for page decisions. Segmentation re-renders the original PDF to retain line detail. Existing valid renders are reused.

# In[ ]:


RENDER_DPI=300
render_rows=[]; render_errors=[]
start=time.time()
for pdf_path_value,pdf_rows in selected.groupby('pdf_path',sort=False):
    pdf_path=Path(pdf_path_value)
    pdf_stem=pdf_rows.iloc[0].pdf.replace('.pdf','')
    patent_dir=PATENT_ROOT/pdf_stem
    page_dir=patent_dir/'drawing_pages'; figure_dir=patent_dir/'figures'; metadata_dir=patent_dir/'metadata'; embedding_dir=patent_dir/'embeddings'
    for folder in [page_dir,figure_dir,metadata_dir,embedding_dir]: folder.mkdir(parents=True,exist_ok=True)
    try:
        with pymupdf.open(pdf_path) as document:
            matrix=pymupdf.Matrix(RENDER_DPI/72.0,RENDER_DPI/72.0)
            for _,row in pdf_rows.iterrows():
                output_path=page_dir/f'page-{int(row.page):04d}.png'
                reused=False
                if output_path.exists():
                    try:
                        with Image.open(output_path) as image: reused=image.width>=1500 and image.height>=1500
                    except Exception: reused=False
                if not reused:
                    document.load_page(int(row.page)-1).get_pixmap(matrix=matrix,alpha=False).save(str(output_path))
                with Image.open(output_path) as image: width,height=image.size
                render_rows.append({'segmentation_page_id':row.segmentation_page_id,'entry_id':row.entry_id,'pdf':row.pdf,
                    'page':int(row.page),'candidate_drawing_probability':float(row.candidate_drawing_probability),
                    'rendered_page_path':str(output_path.resolve()),'render_width':width,'render_height':height,'render_reused':reused})
    except Exception as exc:
        render_errors.append({'pdf':pdf_path.name,'error':repr(exc)})
render_manifest=pd.DataFrame(render_rows)
render_manifest.to_csv(TABLE_ROOT/'rendered_finnish_drawing_pages.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(render_errors,columns=['pdf','error']).to_csv(LOG_ROOT/'render_errors.csv',index=False,encoding='utf-8-sig')
assert len(render_manifest)==len(selected) and not render_errors
print(f'结论：{len(render_manifest)} 页已在每专利目录中以 300 DPI 准备，耗时 {(time.time()-start)/60:.2f} 分钟。')
print(f'Conclusion: {len(render_manifest)} pages are ready at 300 DPI in per-patent folders; elapsed {(time.time()-start)/60:.2f} minutes.')


# ## 第3步：加载仓库 YOLO 模型 / Step 3: Load the repository YOLO model
# 
# 推理参数与瑞典实验保持一致：`imgsz=1024`、`conf=0.5`。裁剪后使用仓库相同的规则，仅保留主要落入检测框的连通墨迹区域。
# 
# Inference matches the Swedish experiment with `imgsz=1024` and `conf=0.5`. Crops use the repository rule that retains only connected ink components primarily contained inside the detection box.

# In[ ]:


IMGSZ=1024; CONFIDENCE=0.5; IOU=0.6; MAX_DET=50
model=YOLO(str(MODEL_PATH))
FALLBACK_CONFIDENCE=0.30
config_text=f'{MODEL_PATH.resolve()}|{MODEL_PATH.stat().st_size}|{IMGSZ}|{CONFIDENCE}|{FALLBACK_CONFIDENCE}|{IOU}|mask50|robust-border-v2'
CONFIG_ID=hashlib.sha256(config_text.encode()).hexdigest()[:16]
print('Model task:',model.task,'| Config ID:',CONFIG_ID)
print('结论：仓库 best.pt 已加载，分割参数和缓存版本已固定。')
print('Conclusion: Repository best.pt is loaded and segmentation parameters and cache version are fixed.')


# ## 第4步：运行 YOLO 检测和连通域遮罩 / Step 4: Run YOLO detection and connected-component masking

# In[ ]:


def process_page(page_row):
    page_path=Path(page_row.rendered_page_path)
    pdf_stem=page_row.pdf.replace('.pdf','')
    patent_dir=PATENT_ROOT/pdf_stem; figure_dir=patent_dir/'figures'; metadata_dir=patent_dir/'metadata'
    metadata_path=metadata_dir/f'page-{int(page_row.page):04d}.json'
    if metadata_path.exists():
        try:
            cached=json.loads(metadata_path.read_text(encoding='utf-8'))
            if cached.get('config_id')==CONFIG_ID and all(Path(item['figure_path']).exists() for item in cached.get('figures',[])):
                return cached['figures'],True
        except Exception: pass
    for stale in figure_dir.glob(f'page-{int(page_row.page):04d}_fig-*.png'): stale.unlink()
    image=cv2.imread(str(page_path))
    if image is None: raise ValueError(f'Cannot read {page_path}')
    height,width=image.shape[:2]; gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    border_width=max(5,min(height,width)//100)
    border_pixels=np.concatenate([gray[:border_width,:].ravel(),gray[-border_width:,:].ravel(),
                                  gray[:, :border_width].ravel(),gray[:, -border_width:].ravel()])
    is_dark=float(np.median(border_pixels))<128
    bg_color=[0,0,0] if is_dark else [255,255,255]
    _,ink=cv2.threshold(gray,128,255,cv2.THRESH_BINARY if is_dark else cv2.THRESH_BINARY_INV)
    _,labels,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=8)
    inference_threshold=CONFIDENCE
    result=model.predict(source=image,imgsz=IMGSZ,conf=inference_threshold,iou=IOU,max_det=MAX_DET,verbose=False,device=device)[0]
    if len(result.boxes)==0:
        inference_threshold=FALLBACK_CONFIDENCE
        result=model.predict(source=image,imgsz=IMGSZ,conf=inference_threshold,iou=IOU,max_det=MAX_DET,verbose=False,device=device)[0]
    figures=[]
    for detection_number,box in enumerate(result.boxes,1):
        x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
        x1=max(0,min(x1,width-1)); y1=max(0,min(y1,height-1)); x2=max(x1+1,min(x2,width)); y2=max(y1+1,min(y2,height))
        crop=image[y1:y2,x1:x2].copy(); label_crop=labels[y1:y2,x1:x2]
        unique_ids,counts=np.unique(label_crop,return_counts=True); valid_ids=[]
        for label_id,count in zip(unique_ids,counts):
            if label_id==0: continue
            total_area=stats[label_id,cv2.CC_STAT_AREA]
            if count/max(1,total_area)>=.5: valid_ids.append(int(label_id))
        mask=np.isin(label_crop,valid_ids); crop[~mask]=bg_color
        output_path=figure_dir/f'page-{int(page_row.page):04d}_fig-{detection_number:03d}.png'
        cv2.imwrite(str(output_path),crop)
        crop_gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        crop_ink=(crop_gray>30) if is_dark else (crop_gray<225)
        class_id=int(box.cls[0]); confidence=float(box.conf[0]); crop_h,crop_w=crop.shape[:2]
        figures.append({'segmentation_page_id':page_row.segmentation_page_id,'entry_id':page_row.entry_id,'pdf':page_row.pdf,
            'page':int(page_row.page),'figure_number':detection_number,'figure_path':str(output_path.resolve()),
            'class_id':class_id,'class_name':str(result.names.get(class_id,class_id)),'confidence':confidence,
            'inference_threshold':inference_threshold,'fallback_detection':bool(inference_threshold<CONFIDENCE),
            'bbox_x1':x1,'bbox_y1':y1,'bbox_x2':x2,'bbox_y2':y2,'width':crop_w,'height':crop_h,
            'bbox_area_ratio':float((crop_w*crop_h)/(width*height)),'ink_ratio':float(crop_ink.mean()),
            'connected_components_retained':len(valid_ids)})
    metadata={'config_id':CONFIG_ID,'page_path':str(page_path.resolve()),'detections':len(figures),'figures':figures}
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return figures,False

all_figures=[]; segmentation_errors=[]; reused_pages=0
stage_start=time.time()
for position,row in render_manifest.iterrows():
    try:
        figures,reused=process_page(row); all_figures.extend(figures); reused_pages+=int(reused)
    except Exception as exc:
        segmentation_errors.append({'segmentation_page_id':row.segmentation_page_id,'pdf':row.pdf,'page':int(row.page),'error':repr(exc)})
    if (position+1)%10==0: print(f'Segmented pages: {position+1}/{len(render_manifest)} | figures: {len(all_figures)}')
figure_manifest=pd.DataFrame(all_figures)
figure_manifest.to_csv(TABLE_ROOT/'finnish_yolo_raw_figures.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(segmentation_errors,columns=['segmentation_page_id','pdf','page','error']).to_csv(LOG_ROOT/'segmentation_errors.csv',index=False,encoding='utf-8-sig')
print(f'结论：151 页 YOLO 分割完成，得到 {len(figure_manifest)} 个原始候选；复用 {reused_pages} 页缓存，错误 {len(segmentation_errors)}，耗时 {(time.time()-stage_start)/60:.2f} 分钟。')
print(f'Conclusion: YOLO completed on 151 pages with {len(figure_manifest)} raw candidates, {reused_pages} cached pages, {len(segmentation_errors)} errors, and {(time.time()-stage_start)/60:.2f} elapsed minutes.')


# ## 第5步：质量检查和 embedding 候选标记 / Step 5: Quality checks and embedding-candidate flags
# 
# 不删除原始 YOLO 结果；只通过尺寸、面积和墨迹比例生成 `keep_for_embedding` 标记，方便后续复核和调整。
# 
# Raw YOLO outputs are never deleted. Size, area, and ink checks only create a `keep_for_embedding` flag for later review and adjustment.

# In[ ]:


if len(figure_manifest):
    figure_manifest['valid_file']=figure_manifest.figure_path.map(lambda p:Path(p).exists() and Path(p).stat().st_size>0)
    figure_manifest['keep_for_embedding']=(figure_manifest.valid_file & (figure_manifest.width>=100) & (figure_manifest.height>=100) &
        (figure_manifest.bbox_area_ratio>=.003) & (figure_manifest.ink_ratio>=.002) & (figure_manifest.ink_ratio<=.80) &
        (figure_manifest.connected_components_retained>=1))
    figure_manifest['quality_reason']=np.where(figure_manifest.keep_for_embedding,'keep',
        np.where(~figure_manifest.valid_file,'invalid_file',
        np.where((figure_manifest.width<100)|(figure_manifest.height<100),'too_small',
        np.where(figure_manifest.bbox_area_ratio<.003,'tiny_page_fraction',
        np.where(figure_manifest.ink_ratio<.002,'almost_blank',
        np.where(figure_manifest.ink_ratio>.80,'too_dense','no_retained_components'))))))
else:
    figure_manifest['keep_for_embedding']=pd.Series(dtype=bool); figure_manifest['quality_reason']=pd.Series(dtype=str)
figure_manifest.to_csv(TABLE_ROOT/'finnish_yolo_figures_manifest.csv',index=False,encoding='utf-8-sig')
detected_page_ids=set(figure_manifest.segmentation_page_id) if len(figure_manifest) else set()
fallback_page_count=int(figure_manifest.loc[figure_manifest.fallback_detection,'segmentation_page_id'].nunique()) if len(figure_manifest) else 0
zero_pages=render_manifest[~render_manifest.segmentation_page_id.isin(detected_page_ids)].copy()
zero_pages.to_csv(TABLE_ROOT/'zero_detection_pages.csv',index=False,encoding='utf-8-sig')
patent_summary=render_manifest.groupby(['entry_id','pdf']).size().rename('drawing_pages').reset_index()
if len(figure_manifest):
    counts=figure_manifest.groupby(['entry_id','pdf']).agg(raw_figures=('figure_path','size'),kept_figures=('keep_for_embedding','sum')).reset_index()
    patent_summary=patent_summary.merge(counts,on=['entry_id','pdf'],how='left')
patent_summary[['raw_figures','kept_figures']]=patent_summary[['raw_figures','kept_figures']].fillna(0).astype(int)
patent_summary.to_csv(TABLE_ROOT/'finnish_yolo_per_patent_summary.csv',index=False,encoding='utf-8-sig')
display(figure_manifest.groupby(['quality_reason']).size().rename('figures').to_frame())
print('Zero-detection pages:',len(zero_pages),'| Fallback pages:',fallback_page_count,'| Patents with kept figures:',int((patent_summary.kept_figures>0).sum()))
print('结论：原始候选已完成文件和几何检查，并标记可进入 embedding 的图形。')
print('Conclusion: Raw candidates passed file and geometry checks and figures eligible for embedding are flagged.')


# ## 第6步：生成分割抽样预览 / Step 6: Generate segmentation sample previews

# In[ ]:


kept=figure_manifest[figure_manifest.keep_for_embedding].copy()
if len(kept):
    preview=kept.sample(min(30,len(kept)),random_state=42)
    columns=5; rows=math.ceil(len(preview)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=(15,3.2*rows)); axes=np.atleast_1d(axes).ravel()
    for ax,(_,item) in zip(axes,preview.iterrows()):
        ax.imshow(Image.open(item.figure_path).convert('RGB'))
        ax.set_title(f'{item.pdf} p{item.page} f{item.figure_number}\nconf={item.confidence:.2f}',fontsize=7); ax.axis('off')
    for ax in axes[len(preview):]: ax.axis('off')
    fig.tight_layout(); preview_path=PREVIEW_ROOT/'finnish_yolo_kept_figures_preview.png'
    fig.savefig(preview_path,dpi=160,bbox_inches='tight'); plt.show()
if len(zero_pages):
    sample_zero=zero_pages.sample(min(12,len(zero_pages)),random_state=42)
    columns=4; rows=math.ceil(len(sample_zero)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=(12,4*rows)); axes=np.atleast_1d(axes).ravel()
    for ax,(_,item) in zip(axes,sample_zero.iterrows()):
        ax.imshow(Image.open(item.rendered_page_path).convert('RGB')); ax.set_title(f'{item.pdf} p{item.page}',fontsize=8); ax.axis('off')
    for ax in axes[len(sample_zero):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(PREVIEW_ROOT/'zero_detection_pages_preview.png',dpi=140,bbox_inches='tight'); plt.show()
print('结论：保留图形和零检测页的抽样预览已经生成，可用于视觉复核。')
print('Conclusion: Sample previews of kept figures and zero-detection pages are generated for visual review.')


# ## 第7步：保存分割总结 / Step 7: Save the segmentation summary

# In[ ]:


summary={'created_at':datetime.now().isoformat(timespec='seconds'),'country':'FI','selected_drawing_pages':int(len(selected)),
    'finnish_pdfs':int(selected.pdf.nunique()),'render_dpi':RENDER_DPI,'yolo_model':str(MODEL_PATH.resolve()),
    'yolo_config_id':CONFIG_ID,'imgsz':IMGSZ,'confidence_threshold':CONFIDENCE,'raw_figures':int(len(figure_manifest)),
    'kept_for_embedding':int(figure_manifest.keep_for_embedding.sum()) if len(figure_manifest) else 0,
    'zero_detection_pages':int(len(zero_pages)),'fallback_detection_pages':int(fallback_page_count),'segmentation_errors':int(len(segmentation_errors)),
    'patents_with_kept_figures':int((patent_summary.kept_figures>0).sum()),
    'figure_manifest':str((TABLE_ROOT/'finnish_yolo_figures_manifest.csv').resolve()),
    'patent_root':str(PATENT_ROOT.resolve())}
(OUTPUT_ROOT/'finland_segmentation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
display(pd.DataFrame([summary]).T.rename(columns={0:'value'}))
print('结论：芬兰 YOLO 分割阶段完成；通过质量标记的图形已可进入 embedding。')
print('Conclusion: Finnish YOLO segmentation is complete and quality-flagged figures are ready for embedding.')


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


# # Part 2 · 使用 DINO 过滤芬兰 YOLO 裁剪 / DINO Filtering of Finnish YOLO Crops
# 
# YOLO 负责定位候选图形，但邮票、签名和图号文字也可能被检测。本 Notebook 使用芬兰 DINOv3 页面分类器作为第二道视觉过滤，并保留所有原始 YOLO 文件。
# 
# YOLO locates candidate figures but can also detect stamps, signatures, and figure-label text. This notebook applies the Finnish DINOv3 page classifier as a second visual filter while preserving every raw YOLO file.

# ## 第0步：加载分割清单和 DINO 分类器 / Step 0: Load the segmentation manifest and DINO classifier

# In[ ]:


from pathlib import Path
from datetime import datetime
import json, math, time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import timm

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm").is_dir()), None)
assert ROOT is not None, "Could not find a parent directory containing embedding_atm."
if not (ROOT/'sample_data').exists(): ROOT=ROOT.parent
SEGMENTATION_ROOT=ROOT/'embedding_atm'/'finland_segmentation'
INPUT_MANIFEST=SEGMENTATION_ROOT/'tables'/'finnish_yolo_figures_manifest.csv'
CLASSIFIER_PATH=ROOT/'embedding_atm'/'finnish_page_classifier_v1'/'models'/'finnish_page_classifier_dinov3_candidate_v1.joblib'
OUTPUT_ROOT=ROOT/'embedding_atm'/'finland_figure_filter'
TABLE_ROOT=OUTPUT_ROOT/'tables'; PREVIEW_ROOT=OUTPUT_ROOT/'previews'
for folder in [OUTPUT_ROOT,TABLE_ROOT,PREVIEW_ROOT]: folder.mkdir(parents=True,exist_ok=True)
manifest=pd.read_csv(INPUT_MANIFEST)
classifier_bundle=joblib.load(CLASSIFIER_PATH)
assert classifier_bundle['feature_variant']=='dino_only'
assert manifest.figure_path.map(lambda p:Path(p).exists()).all()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('YOLO crops:',len(manifest),'| Device:',device,'| Page-classifier threshold:',classifier_bundle['decision_threshold'])
print('结论：519 个 YOLO 裁剪和芬兰 DINO 分类器已加载。')
print('Conclusion: The 519 YOLO crops and Finnish DINO classifier are loaded.')


# ## 第1步：提取裁剪 DINOv3 embedding / Step 1: Extract DINOv3 embeddings for the crops

# In[ ]:


MODEL_NAME=classifier_bundle['dino_model']; INPUT_SIZE=256
model=timm.create_model(MODEL_NAME,pretrained=True,num_classes=0).to(device).eval()
embedding_dim=int(model.num_features)
mean=torch.tensor([.485,.456,.406]).view(3,1,1); std=torch.tensor([.229,.224,.225]).view(3,1,1)
def crop_tensor(path):
    image=Image.open(path).convert('RGB'); inner=int(INPUT_SIZE*.92)
    contained=ImageOps.contain(image,(inner,inner),Image.Resampling.BICUBIC)
    canvas=Image.new('RGB',(INPUT_SIZE,INPUT_SIZE),'white')
    canvas.paste(contained,((INPUT_SIZE-contained.width)//2,(INPUT_SIZE-contained.height)//2))
    array=np.asarray(canvas,dtype=np.float32)/255.0
    return (torch.from_numpy(array).permute(2,0,1)-mean)/std
embedding_path=OUTPUT_ROOT/'dinov3_yolo_crop_embeddings.npy'
key_path=TABLE_ROOT/'dinov3_yolo_crop_embedding_manifest.csv'
keys=[f'{Path(p).resolve()}|{Path(p).stat().st_size}' for p in manifest.figure_path]
cache_valid=False
if embedding_path.exists() and key_path.exists():
    cached=np.load(embedding_path); cached_keys=pd.read_csv(key_path).crop_key.tolist()
    cache_valid=cached.shape==(len(manifest),embedding_dim) and cached_keys==keys
if cache_valid:
    embeddings=cached.astype(np.float32); print('Reused cached crop embeddings.')
else:
    outputs=[]; batch_size=6 if device.type=='cpu' else 24; start=time.time()
    with torch.inference_mode():
        for batch_start in range(0,len(manifest),batch_size):
            paths=manifest.figure_path.iloc[batch_start:batch_start+batch_size]
            batch=torch.stack([crop_tensor(path) for path in paths]).to(device)
            vectors=model(batch); vectors=vectors[0] if isinstance(vectors,(tuple,list)) else vectors
            outputs.append(F.normalize(vectors.float(),p=2,dim=1).cpu().numpy())
        embeddings=np.concatenate(outputs).astype(np.float32)
    np.save(embedding_path,embeddings)
    pd.DataFrame({'crop_index':np.arange(len(manifest)),'crop_key':keys}).to_csv(key_path,index=False,encoding='utf-8-sig')
    print(f'Embedding time: {(time.time()-start)/60:.2f} minutes')
assert embeddings.shape==(len(manifest),embedding_dim)
print('结论：所有 YOLO 裁剪已转换为 768 维 DINOv3 embedding。')
print('Conclusion: Every YOLO crop is converted into a 768-dimensional DINOv3 embedding.')


# ## 第2步：计算技术图概率并生成最终标记 / Step 2: Score technical-drawing probability and create final flags

# In[ ]:


visual_probability=classifier_bundle['model'].predict_proba(embeddings)[:,1]
PAGE_CLASSIFIER_THRESHOLD=float(classifier_bundle['decision_threshold'])
CROP_DINO_MINIMUM=0.10
STAMP_MAXIMUM_PAGE_FRACTION=0.04
STAMP_MINIMUM_INK_RATIO=0.14
STAMP_MAXIMUM_DINO_PROBABILITY=0.99
STAMP_MINIMUM_COMPONENT_DENSITY_PER_10K=5.0
scored=manifest.copy()
scored['dino_drawing_probability']=visual_probability
scored['passes_dino_filter']=scored.dino_drawing_probability>=CROP_DINO_MINIMUM
scored['component_density_per_10k']=scored.connected_components_retained/(scored.width*scored.height)*10000
scored['stamp_like']=((scored.bbox_area_ratio<=STAMP_MAXIMUM_PAGE_FRACTION)&(scored.ink_ratio>=STAMP_MINIMUM_INK_RATIO)&
    (scored.dino_drawing_probability<STAMP_MAXIMUM_DINO_PROBABILITY)&
    (scored.component_density_per_10k>=STAMP_MINIMUM_COMPONENT_DENSITY_PER_10K))
scored['embedding_ready']=(scored.keep_for_embedding.astype(bool)&scored.passes_dino_filter&~scored.stamp_like)
scored['final_filter_reason']=np.where(~scored.keep_for_embedding.astype(bool),scored.quality_reason,
    np.where(scored.stamp_like,'stamp_like',
    np.where(~scored.passes_dino_filter,'dino_non_drawing','embedding_ready')))
scored['manual_filter_override']=''
manual_keep=((scored.pdf=='FI.819.A.pdf')&(scored.page==8)&(scored.figure_number==2))
assert int(manual_keep.sum())==1
scored.loc[manual_keep,'embedding_ready']=True
scored.loc[manual_keep,'final_filter_reason']='manual_keep_override'
scored.loc[manual_keep,'manual_filter_override']='visual_confirmed_mechanical_figure'
scored.to_csv(TABLE_ROOT/'finnish_figures_embedding_ready_manifest.csv',index=False,encoding='utf-8-sig')
display(scored.groupby('final_filter_reason').size().rename('figures').to_frame())
print('DINO probability quantiles:',scored.dino_drawing_probability.quantile([0,.1,.25,.5,.75,.9,1]).round(4).to_dict())
print('结论：几何质量与 DINO 技术图概率已经合并为 embedding_ready 标记。')
print('Conclusion: Geometry quality and DINO technical-drawing probability are combined into the embedding_ready flag.')


# ## 第3步：检查保留与拒绝样本 / Step 3: Review accepted and rejected samples

# In[ ]:


def save_preview(frame,path,title,limit=30):
    subset=frame.head(limit); columns=5; rows=max(1,math.ceil(len(subset)/columns))
    fig,axes=plt.subplots(rows,columns,figsize=(15,3.2*rows)); axes=np.atleast_1d(axes).ravel()
    for ax,(_,item) in zip(axes,subset.iterrows()):
        ax.imshow(Image.open(item.figure_path).convert('RGB'))
        ax.set_title(f'{item.pdf} p{item.page} f{item.figure_number}\nDINO={item.dino_drawing_probability:.3f}',fontsize=7); ax.axis('off')
    for ax in axes[len(subset):]: ax.axis('off')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=160,bbox_inches='tight'); plt.close(fig)
accepted=scored[scored.embedding_ready].sample(min(30,int(scored.embedding_ready.sum())),random_state=42)
rejected=scored[~scored.embedding_ready].sort_values(['final_filter_reason','dino_drawing_probability'])
save_preview(accepted,PREVIEW_ROOT/'accepted_finnish_figures.png','Accepted for embedding')
save_preview(rejected,PREVIEW_ROOT/'rejected_non_drawing_crops.png','Rejected by DINO')
display(Image.open(PREVIEW_ROOT/'rejected_non_drawing_crops.png'))
print('结论：DINO 接受与拒绝样本预览已经生成，原始文件没有删除。')
print('Conclusion: DINO accepted and rejected sample previews are generated without deleting raw files.')


# ## 第4步：保存过滤总结 / Step 4: Save the filtering summary

# In[ ]:


summary={'created_at':datetime.now().isoformat(timespec='seconds'),'raw_yolo_figures':int(len(scored)),
    'geometry_kept':int(scored.keep_for_embedding.sum()),'page_classifier_threshold':PAGE_CLASSIFIER_THRESHOLD,
    'crop_dino_minimum':CROP_DINO_MINIMUM,
    'stamp_maximum_page_fraction':STAMP_MAXIMUM_PAGE_FRACTION,'stamp_minimum_ink_ratio':STAMP_MINIMUM_INK_RATIO,
    'stamp_maximum_dino_probability':STAMP_MAXIMUM_DINO_PROBABILITY,
    'stamp_minimum_component_density_per_10k':STAMP_MINIMUM_COMPONENT_DENSITY_PER_10K,
    'embedding_ready':int(scored.embedding_ready.sum()),
    'dino_rejected':int((scored.final_filter_reason=='dino_non_drawing').sum()),
    'stamp_like_rejected':int((scored.final_filter_reason=='stamp_like').sum()),
    'manual_keep_overrides':int((scored.final_filter_reason=='manual_keep_override').sum()),
    'source_files_preserved':True,'model':classifier_bundle['dino_model'],
    'final_manifest':str((TABLE_ROOT/'finnish_figures_embedding_ready_manifest.csv').resolve())}
(OUTPUT_ROOT/'figure_filter_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
display(pd.DataFrame([summary]).T.rename(columns={0:'value'}))
print('结论：芬兰图形过滤完成，embedding_ready 清单可作为下一步 embedding 输入。')
print('Conclusion: Finnish figure filtering is complete and the embedding_ready manifest is the input for the next embedding stage.')


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


# # Part 3 · 芬兰专利图 DINOv3 Embedding / DINOv3 Embeddings for Finnish Patent Figures
# 
# 本 Notebook 将最终 `embedding_ready` 芬兰图形写入每专利 embedding 文件夹，并建立可用于瑞典—芬兰检索的芬兰全局索引。已在过滤阶段计算的 DINOv3 向量会被严格校验并复用。
# 
# This notebook writes every final `embedding_ready` Finnish figure into its per-patent embedding folder and builds the global Finnish index for Swedish-Finnish retrieval. DINOv3 vectors computed during filtering are validated and reused.

# ## 第0步：加载最终图形清单和缓存向量 / Step 0: Load the final figure manifest and cached vectors

# In[ ]:


from pathlib import Path
from datetime import datetime
import json, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm").is_dir()), None)
assert ROOT is not None, "Could not find a parent directory containing embedding_atm."
if not (ROOT/'sample_data').exists(): ROOT=ROOT.parent
FILTER_ROOT=ROOT/'embedding_atm'/'finland_figure_filter'
SEGMENTATION_ROOT=ROOT/'embedding_atm'/'finland_segmentation'
SOURCE_MANIFEST=FILTER_ROOT/'tables'/'finnish_figures_embedding_ready_manifest.csv'
SOURCE_EMBEDDINGS=FILTER_ROOT/'dinov3_yolo_crop_embeddings.npy'
SOURCE_KEYS=FILTER_ROOT/'tables'/'dinov3_yolo_crop_embedding_manifest.csv'
OUTPUT_ROOT=ROOT/'embedding_atm'/'cross_country_retrieval'
INDEX_ROOT=OUTPUT_ROOT/'indexes'/'finland'; TABLE_ROOT=OUTPUT_ROOT/'tables'/'finland'; PREVIEW_ROOT=OUTPUT_ROOT/'previews'/'finland'
for folder in [OUTPUT_ROOT,INDEX_ROOT,TABLE_ROOT,PREVIEW_ROOT]: folder.mkdir(parents=True,exist_ok=True)
full_manifest=pd.read_csv(SOURCE_MANIFEST)
all_embeddings=np.load(SOURCE_EMBEDDINGS).astype(np.float32)
key_manifest=pd.read_csv(SOURCE_KEYS)
assert len(full_manifest)==len(all_embeddings)==len(key_manifest)
expected_keys=[f'{Path(p).resolve()}|{Path(p).stat().st_size}' for p in full_manifest.figure_path]
assert expected_keys==key_manifest.crop_key.tolist(), 'Embedding cache order does not match the figure manifest.'
selected=full_manifest[full_manifest.embedding_ready.astype(bool)].copy()
selected_indices=selected.index.to_numpy(dtype=int)
embeddings=all_embeddings[selected_indices].copy()
assert len(selected)==486 and embeddings.shape==(486,768)
norms=np.linalg.norm(embeddings,axis=1)
assert np.allclose(norms,1.0,atol=1e-5)
print('Figures:',len(selected),'| Patents:',selected.pdf.nunique(),'| Shape:',embeddings.shape)
print('结论：486 张最终芬兰图形与其 768 维归一化 DINOv3 向量已严格对齐。')
print('Conclusion: The 486 final Finnish figures are strictly aligned with their normalized 768-dimensional DINOv3 vectors.')


# ## 第1步：保存每专利 embedding / Step 1: Save per-patent embeddings

# In[ ]:


MODEL_SLUG='dinov3_vitb16'; MODEL_NAME='vit_base_patch16_dinov3.lvd1689m'
MODEL_DIMENSION=embeddings.shape[1]
selected=selected.reset_index(names='source_manifest_index')
embedding_rows=[]; expected_by_dir={}
for position,row in selected.iterrows():
    pdf_stem=row.pdf.replace('.pdf','')
    embedding_dir=SEGMENTATION_ROOT/'patents'/pdf_stem/'embeddings'/MODEL_SLUG
    embedding_dir.mkdir(parents=True,exist_ok=True)
    output_path=embedding_dir/f'{Path(row.figure_path).stem}.npy'
    np.save(output_path,embeddings[position])
    expected_by_dir.setdefault(embedding_dir,set()).add(output_path.name)
    embedding_rows.append({'finland_figure_id':f'FI-FIG-{position+1:04d}','global_index':position,
        'entry_id':row.entry_id,'pdf':row.pdf,'page':int(row.page),'figure_number':int(row.figure_number),
        'figure_path':str(Path(row.figure_path).resolve()),'embedding_path':str(output_path.resolve()),
        'model':MODEL_NAME,'embedding_dimension':MODEL_DIMENSION,
        'yolo_confidence':float(row.confidence),'dino_filter_probability':float(row.dino_drawing_probability),
        'manual_filter_override':row.manual_filter_override if pd.notna(row.manual_filter_override) else ''})
for embedding_dir,expected_names in expected_by_dir.items():
    for stale in embedding_dir.glob('*.npy'):
        if stale.name not in expected_names: stale.unlink()
embedding_manifest=pd.DataFrame(embedding_rows)
embedding_manifest.to_csv(INDEX_ROOT/f'{MODEL_SLUG}_manifest.csv',index=False,encoding='utf-8-sig')
print('Per-patent embedding files:',sum(1 for p in expected_by_dir for _ in p.glob('*.npy')))
print('结论：486 个向量已写入 90 个含有效图的芬兰专利目录，并清除了本模型目录中的过期向量。')
print('Conclusion: All 486 vectors are saved under 90 Finnish patent folders containing valid figures, and stale vectors for this model are removed.')


# ## 第2步：建立芬兰全局检索索引 / Step 2: Build the global Finnish retrieval index

# In[ ]:


matrix_path=INDEX_ROOT/f'{MODEL_SLUG}_embeddings.npy'
bundle_path=INDEX_ROOT/f'{MODEL_SLUG}_index_bundle.npz'
np.save(matrix_path,embeddings)
np.savez_compressed(bundle_path,embeddings=embeddings,
    finland_figure_ids=embedding_manifest.finland_figure_id.to_numpy(),
    entry_ids=embedding_manifest.entry_id.to_numpy(),pdfs=embedding_manifest.pdf.to_numpy(),
    pages=embedding_manifest.page.to_numpy(),figure_paths=embedding_manifest.figure_path.to_numpy())
index_info={'created_at':datetime.now().isoformat(timespec='seconds'),'country':'FI','figures':len(embedding_manifest),
    'patents':int(embedding_manifest.pdf.nunique()),'model':MODEL_NAME,'dimension':MODEL_DIMENSION,
    'normalized':True,'embedding_matrix':str(matrix_path.resolve()),
    'embedding_manifest':str((INDEX_ROOT/f'{MODEL_SLUG}_manifest.csv').resolve()),
    'index_bundle':str(bundle_path.resolve())}
(INDEX_ROOT/f'{MODEL_SLUG}_index_info.json').write_text(json.dumps(index_info,indent=2),encoding='utf-8')
display(pd.DataFrame([index_info]).T.rename(columns={0:'value'}))
print('结论：芬兰全局矩阵、清单和压缩索引包已经生成。')
print('Conclusion: The global Finnish matrix, manifest, and compressed index bundle are generated.')


# ## 第3步：生成芬兰内部 Top-5 邻居 / Step 3: Generate Finnish internal Top-5 neighbours

# In[ ]:


similarity=embeddings@embeddings.T
np.fill_diagonal(similarity,-np.inf)
top_k=5; neighbour_indices=np.argsort(-similarity,axis=1)[:,:top_k]
neighbour_rows=[]
for query_index,neighbours in enumerate(neighbour_indices):
    query=embedding_manifest.iloc[query_index]
    for rank,neighbour_index in enumerate(neighbours,1):
        neighbour=embedding_manifest.iloc[int(neighbour_index)]
        neighbour_rows.append({'query_figure_id':query.finland_figure_id,'query_pdf':query.pdf,
            'query_page':int(query.page),'query_figure_path':query.figure_path,'rank':rank,
            'neighbour_figure_id':neighbour.finland_figure_id,'neighbour_pdf':neighbour.pdf,
            'neighbour_page':int(neighbour.page),'neighbour_figure_path':neighbour.figure_path,
            'cosine_similarity':float(similarity[query_index,neighbour_index]),
            'same_patent':bool(query.pdf==neighbour.pdf)})
neighbours=pd.DataFrame(neighbour_rows)
neighbours.to_csv(TABLE_ROOT/'finland_top5_internal_neighbours.csv',index=False,encoding='utf-8-sig')
display(pd.DataFrame([{'mean_top1_similarity':neighbours[neighbours['rank']==1].cosine_similarity.mean(),
    'top5_same_patent_rate':neighbours.same_patent.mean()}]).round(4))
print('结论：芬兰内部 Top-5 邻居表已生成，可用于发现重复图和剩余非技术图候选。')
print('Conclusion: The Finnish internal Top-5 neighbour table is ready for duplicate and remaining nuisance review.')


# ## 第4步：生成 Top-3 检索预览 / Step 4: Generate a Top-3 retrieval preview

# In[ ]:


query_indices=np.linspace(0,len(embedding_manifest)-1,12,dtype=int)
fig,axes=plt.subplots(len(query_indices),4,figsize=(14,3.2*len(query_indices)))
for row_number,query_index in enumerate(query_indices):
    query=embedding_manifest.iloc[query_index]
    axes[row_number,0].imshow(Image.open(query.figure_path).convert('RGB'))
    axes[row_number,0].set_title(f'QUERY {query.finland_figure_id}\n{query.pdf} p{query.page}',fontsize=7); axes[row_number,0].axis('off')
    for column,neighbour_index in enumerate(neighbour_indices[query_index,:3],1):
        neighbour=embedding_manifest.iloc[int(neighbour_index)]
        axes[row_number,column].imshow(Image.open(neighbour.figure_path).convert('RGB'))
        axes[row_number,column].set_title(f'Top {column}: {neighbour.finland_figure_id}\ncos={similarity[query_index,neighbour_index]:.3f}',fontsize=7)
        axes[row_number,column].axis('off')
fig.tight_layout(); preview_path=PREVIEW_ROOT/'finland_top3_internal_retrieval.png'
fig.savefig(preview_path,dpi=150,bbox_inches='tight'); plt.show()
print('结论：12 个芬兰查询的 Top-3 检索预览已经保存。')
print('Conclusion: A Top-3 retrieval preview for 12 Finnish queries is saved.')


# ## 第5步：保存芬兰 embedding 总结 / Step 5: Save the Finnish embedding summary

# In[ ]:


summary={**index_info,'per_patent_embedding_files':int(len(embedding_manifest)),
    'mean_top1_internal_similarity':float(neighbours[neighbours['rank']==1].cosine_similarity.mean()),
    'top5_same_patent_rate':float(neighbours.same_patent.mean()),
    'top5_neighbour_table':str((TABLE_ROOT/'finland_top5_internal_neighbours.csv').resolve()),
    'preview':str(preview_path.resolve())}
(OUTPUT_ROOT/'finland_embedding_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
display(pd.DataFrame([summary]).T.rename(columns={0:'value'}))
print('结论：芬兰 486 图 embedding 和索引阶段完成，可作为跨国检索的芬兰图库。')
print('Conclusion: Finnish embedding and indexing for 486 figures is complete and ready as the Finnish cross-country gallery.')


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


# # Part 4 · 配对瑞典专利图页 YOLO 分割 / YOLO Segmentation of Matched Swedish Patent Drawing Pages
# 
# 本 Notebook 复用已经验证过的瑞典页面分类结果，只处理被判定为 `drawing` 的页面。输出按 `entry_id / patent` 分开保存，因此即使两个配对条目使用相同的瑞典文件名也不会互相覆盖。所有表格字段和值保持英文。
# 
# This notebook reuses the validated Swedish page-classifier predictions and processes only pages labelled `drawing`. Outputs are separated by `entry_id / patent`, so duplicate Swedish filenames from different matched entries cannot overwrite each other. All table fields and values remain English-only.

# ## 第0步：加载依赖并固定路径 / Step 0: Load dependencies and freeze paths

# In[ ]:


from pathlib import Path
from datetime import datetime
import hashlib, importlib.metadata, json, math, time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image
import torch
from ultralytics import YOLO

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm").is_dir()), None)
assert ROOT is not None, "Could not find a parent directory containing embedding_atm."
if not (ROOT/'Swe-Fin-matched-patents').exists(): ROOT=ROOT.parent
REPO=ROOT/'swe_pre_process_pipeline-main'/'swe_pre_process_pipeline-main'
MODEL_PATH=REPO/'artifacts'/'best.pt'
PAGE_MANIFEST_PATH=ROOT/'embedding_atm'/'finnish_page_classifier_pilot'/'tables'/'page_manifest_with_old_predictions.csv'
OUTPUT_ROOT=ROOT/'embedding_atm'/'sweden_matched_segmentation'
PATENT_ROOT=OUTPUT_ROOT/'patents'; TABLE_ROOT=OUTPUT_ROOT/'tables'; PREVIEW_ROOT=OUTPUT_ROOT/'previews'; LOG_ROOT=OUTPUT_ROOT/'logs'
for folder in [OUTPUT_ROOT,PATENT_ROOT,TABLE_ROOT,PREVIEW_ROOT,LOG_ROOT]: folder.mkdir(parents=True,exist_ok=True)
assert MODEL_PATH.exists(),MODEL_PATH
assert PAGE_MANIFEST_PATH.exists(),PAGE_MANIFEST_PATH
device=0 if torch.cuda.is_available() else 'cpu'
print('Torch:',torch.__version__,'| Ultralytics:',importlib.metadata.version('ultralytics'),'| Device:',device)
print('结论：瑞典页面分类清单、仓库 YOLO 权重和输出目录已准备。')
print('Conclusion: The Swedish page-classification manifest, repository YOLO weights, and output folders are ready.')


# ## 第1步：选择瑞典 drawing pages / Step 1: Select Swedish drawing pages

# In[ ]:


all_pages=pd.read_csv(PAGE_MANIFEST_PATH)
selected=all_pages[(all_pages.country=='SE') & (all_pages.old_predicted_page_type=='drawing')].copy()
selected=selected.sort_values(['entry_id','pdf','page']).reset_index(drop=True)
selected.insert(0,'segmentation_page_id',[f'SE-DRAW-{i+1:04d}' for i in range(len(selected))])
assert len(selected)==214,f'Expected 214 Swedish drawing pages, found {len(selected)}'
assert selected.pdf_path.map(lambda p:Path(p).exists()).all()
selected.to_csv(TABLE_ROOT/'selected_swedish_drawing_pages.csv',index=False,encoding='utf-8-sig')
display(selected.groupby('entry_id').size().describe().to_frame('drawing_pages_per_entry'))
print('Drawing pages:',len(selected),'| PDF files:',len(selected.pdf_path.unique()),'| Entries:',selected.entry_id.nunique(),'| Filename values:',selected.pdf.nunique())
print('结论：214 个瑞典 drawing pages 已固定；后续始终以 entry_id 和完整路径区分文件。')
print('Conclusion: The 214 Swedish drawing pages are frozen, and later stages distinguish files by entry_id and full path.')


# ## 第2步：按配对条目渲染 300 DPI 图页 / Step 2: Render drawing pages at 300 DPI by matched entry

# In[ ]:


RENDER_DPI=300
render_rows=[]; render_errors=[]; stage_start=time.time()
for (entry_id,pdf_path_value),pdf_rows in selected.groupby(['entry_id','pdf_path'],sort=False):
    pdf_path=Path(pdf_path_value); pdf_stem=pdf_path.stem
    patent_dir=PATENT_ROOT/entry_id/pdf_stem
    page_dir=patent_dir/'drawing_pages'; figure_dir=patent_dir/'figures'; metadata_dir=patent_dir/'metadata'; embedding_dir=patent_dir/'embeddings'
    for folder in [page_dir,figure_dir,metadata_dir,embedding_dir]: folder.mkdir(parents=True,exist_ok=True)
    try:
        with pymupdf.open(pdf_path) as document:
            matrix=pymupdf.Matrix(RENDER_DPI/72.0,RENDER_DPI/72.0)
            for _,row in pdf_rows.iterrows():
                output_path=page_dir/f'page-{int(row.page):04d}.png'; reused=False
                if output_path.exists():
                    try:
                        with Image.open(output_path) as image: reused=image.width>=1500 and image.height>=1500
                    except Exception: reused=False
                if not reused: document.load_page(int(row.page)-1).get_pixmap(matrix=matrix,alpha=False).save(str(output_path))
                with Image.open(output_path) as image: width,height=image.size
                render_rows.append({'segmentation_page_id':row.segmentation_page_id,'entry_id':entry_id,'pdf':row.pdf,'pdf_path':str(pdf_path.resolve()),
                    'page':int(row.page),'drawing_probability':float(row.old_drawing_probability),'rendered_page_path':str(output_path.resolve()),
                    'render_width':width,'render_height':height,'render_reused':reused})
    except Exception as exc: render_errors.append({'entry_id':entry_id,'pdf':pdf_path.name,'error':repr(exc)})
render_manifest=pd.DataFrame(render_rows)
render_manifest.to_csv(TABLE_ROOT/'rendered_swedish_drawing_pages.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(render_errors,columns=['entry_id','pdf','error']).to_csv(LOG_ROOT/'render_errors.csv',index=False,encoding='utf-8-sig')
assert len(render_manifest)==len(selected) and not render_errors
print(f'结论：{len(render_manifest)} 个瑞典图页已按配对条目保存为 300 DPI；耗时 {(time.time()-stage_start)/60:.2f} 分钟。')
print(f'Conclusion: {len(render_manifest)} Swedish drawing pages are saved at 300 DPI by matched entry; elapsed {(time.time()-stage_start)/60:.2f} minutes.')


# ## 第3步：加载仓库 YOLO 并固定缓存版本 / Step 3: Load repository YOLO and freeze the cache version

# In[ ]:


IMGSZ=1024; CONFIDENCE=0.50; FALLBACK_CONFIDENCE=0.30; IOU=0.60; MAX_DET=50
model=YOLO(str(MODEL_PATH))
config_text=f'{MODEL_PATH.resolve()}|{MODEL_PATH.stat().st_size}|{IMGSZ}|{CONFIDENCE}|{FALLBACK_CONFIDENCE}|{IOU}|mask50|robust-border-v2|entry-aware'
CONFIG_ID=hashlib.sha256(config_text.encode()).hexdigest()[:16]
print('Model task:',model.task,'| Config ID:',CONFIG_ID)
print('结论：仓库 best.pt 已加载，推理阈值和 entry-aware 缓存版本已固定。')
print('Conclusion: Repository best.pt is loaded, and inference thresholds plus the entry-aware cache version are fixed.')


# ## 第4步：YOLO 分割并保留图内连通区域 / Step 4: Segment with YOLO and retain in-box connected components

# In[ ]:


def process_page(page_row):
    page_path=Path(page_row.rendered_page_path); pdf_stem=Path(page_row.pdf).stem
    patent_dir=PATENT_ROOT/page_row.entry_id/pdf_stem; figure_dir=patent_dir/'figures'; metadata_dir=patent_dir/'metadata'
    metadata_path=metadata_dir/f'page-{int(page_row.page):04d}.json'
    if metadata_path.exists():
        try:
            cached=json.loads(metadata_path.read_text(encoding='utf-8'))
            if cached.get('config_id')==CONFIG_ID and all(Path(item['figure_path']).exists() for item in cached.get('figures',[])): return cached['figures'],True
        except Exception: pass
    for stale in figure_dir.glob(f'page-{int(page_row.page):04d}_fig-*.png'): stale.unlink()
    image=cv2.imread(str(page_path))
    if image is None: raise ValueError(f'Cannot read {page_path}')
    height,width=image.shape[:2]; gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); border_width=max(5,min(height,width)//100)
    border_pixels=np.concatenate([gray[:border_width,:].ravel(),gray[-border_width:,:].ravel(),gray[:,:border_width].ravel(),gray[:,-border_width:].ravel()])
    is_dark=float(np.median(border_pixels))<128; bg_color=[0,0,0] if is_dark else [255,255,255]
    _,ink=cv2.threshold(gray,128,255,cv2.THRESH_BINARY if is_dark else cv2.THRESH_BINARY_INV)
    _,labels,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=8)
    inference_threshold=CONFIDENCE
    result=model.predict(source=image,imgsz=IMGSZ,conf=inference_threshold,iou=IOU,max_det=MAX_DET,verbose=False,device=device)[0]
    if len(result.boxes)==0:
        inference_threshold=FALLBACK_CONFIDENCE
        result=model.predict(source=image,imgsz=IMGSZ,conf=inference_threshold,iou=IOU,max_det=MAX_DET,verbose=False,device=device)[0]
    figures=[]
    for detection_number,box in enumerate(result.boxes,1):
        x1,y1,x2,y2=map(int,box.xyxy[0].tolist()); x1=max(0,min(x1,width-1)); y1=max(0,min(y1,height-1)); x2=max(x1+1,min(x2,width)); y2=max(y1+1,min(y2,height))
        crop=image[y1:y2,x1:x2].copy(); label_crop=labels[y1:y2,x1:x2]
        unique_ids,counts=np.unique(label_crop,return_counts=True); valid_ids=[]
        for label_id,count in zip(unique_ids,counts):
            if label_id==0: continue
            if count/max(1,stats[label_id,cv2.CC_STAT_AREA])>=0.5: valid_ids.append(int(label_id))
        mask=np.isin(label_crop,valid_ids); crop[~mask]=bg_color
        output_path=figure_dir/f'page-{int(page_row.page):04d}_fig-{detection_number:03d}.png'; cv2.imwrite(str(output_path),crop)
        crop_gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY); crop_ink=(crop_gray>30) if is_dark else (crop_gray<225)
        class_id=int(box.cls[0]); crop_h,crop_w=crop.shape[:2]
        figures.append({'segmentation_page_id':page_row.segmentation_page_id,'entry_id':page_row.entry_id,'pdf':page_row.pdf,'pdf_path':page_row.pdf_path,
            'page':int(page_row.page),'figure_number':detection_number,'figure_path':str(output_path.resolve()),'class_id':class_id,
            'class_name':str(result.names.get(class_id,class_id)),'confidence':float(box.conf[0]),'inference_threshold':inference_threshold,
            'fallback_detection':bool(inference_threshold<CONFIDENCE),'bbox_x1':x1,'bbox_y1':y1,'bbox_x2':x2,'bbox_y2':y2,
            'width':crop_w,'height':crop_h,'bbox_area_ratio':float((crop_w*crop_h)/(width*height)),'ink_ratio':float(crop_ink.mean()),
            'connected_components_retained':len(valid_ids)})
    metadata_path.write_text(json.dumps({'config_id':CONFIG_ID,'page_path':str(page_path.resolve()),'detections':len(figures),'figures':figures},indent=2),encoding='utf-8')
    return figures,False

all_figures=[]; segmentation_errors=[]; reused_pages=0; stage_start=time.time()
for position,row in render_manifest.iterrows():
    try:
        figures,reused=process_page(row); all_figures.extend(figures); reused_pages+=int(reused)
    except Exception as exc: segmentation_errors.append({'segmentation_page_id':row.segmentation_page_id,'entry_id':row.entry_id,'pdf':row.pdf,'page':int(row.page),'error':repr(exc)})
    if (position+1)%20==0: print(f'Segmented pages: {position+1}/{len(render_manifest)} | figures: {len(all_figures)}')
figure_manifest=pd.DataFrame(all_figures)
figure_manifest.to_csv(TABLE_ROOT/'swedish_yolo_raw_figures.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(segmentation_errors,columns=['segmentation_page_id','entry_id','pdf','page','error']).to_csv(LOG_ROOT/'segmentation_errors.csv',index=False,encoding='utf-8-sig')
print(f'结论：{len(render_manifest)} 页完成 YOLO 分割，得到 {len(figure_manifest)} 个原始候选；复用 {reused_pages} 页缓存，错误 {len(segmentation_errors)}，耗时 {(time.time()-stage_start)/60:.2f} 分钟。')
print(f'Conclusion: YOLO completed on {len(render_manifest)} pages with {len(figure_manifest)} raw candidates, {reused_pages} cached pages, {len(segmentation_errors)} errors, and {(time.time()-stage_start)/60:.2f} elapsed minutes.')


# ## 第5步：执行基础质量检查 / Step 5: Apply basic quality checks

# In[ ]:


if len(figure_manifest):
    figure_manifest['valid_file']=figure_manifest.figure_path.map(lambda p:Path(p).exists() and Path(p).stat().st_size>0)
    figure_manifest['keep_for_embedding']=(figure_manifest.valid_file & (figure_manifest.width>=100) & (figure_manifest.height>=100) &
        (figure_manifest.bbox_area_ratio>=0.003) & (figure_manifest.ink_ratio>=0.002) & (figure_manifest.ink_ratio<=0.80) &
        (figure_manifest.connected_components_retained>=1))
    figure_manifest['quality_reason']=np.where(figure_manifest.keep_for_embedding,'keep',np.where(~figure_manifest.valid_file,'invalid_file',
        np.where((figure_manifest.width<100)|(figure_manifest.height<100),'too_small',np.where(figure_manifest.bbox_area_ratio<0.003,'tiny_page_fraction',
        np.where(figure_manifest.ink_ratio<0.002,'almost_blank',np.where(figure_manifest.ink_ratio>0.80,'too_dense','no_retained_components'))))))
else:
    figure_manifest['keep_for_embedding']=pd.Series(dtype=bool); figure_manifest['quality_reason']=pd.Series(dtype=str)
figure_manifest.to_csv(TABLE_ROOT/'swedish_yolo_figures_manifest.csv',index=False,encoding='utf-8-sig')
detected_ids=set(figure_manifest.segmentation_page_id) if len(figure_manifest) else set()
zero_pages=render_manifest[~render_manifest.segmentation_page_id.isin(detected_ids)].copy(); zero_pages.to_csv(TABLE_ROOT/'zero_detection_pages.csv',index=False,encoding='utf-8-sig')
fallback_pages=int(figure_manifest.loc[figure_manifest.fallback_detection,'segmentation_page_id'].nunique()) if len(figure_manifest) else 0
patent_summary=render_manifest.groupby(['entry_id','pdf']).size().rename('drawing_pages').reset_index()
if len(figure_manifest):
    counts=figure_manifest.groupby(['entry_id','pdf']).agg(raw_figures=('figure_path','size'),kept_figures=('keep_for_embedding','sum')).reset_index()
    patent_summary=patent_summary.merge(counts,on=['entry_id','pdf'],how='left')
patent_summary[['raw_figures','kept_figures']]=patent_summary[['raw_figures','kept_figures']].fillna(0).astype(int)
patent_summary.to_csv(TABLE_ROOT/'swedish_yolo_per_entry_summary.csv',index=False,encoding='utf-8-sig')
display(figure_manifest.groupby('quality_reason').size().rename('figures').to_frame())
print('Zero-detection pages:',len(zero_pages),'| Fallback pages:',fallback_pages,'| Entries with kept figures:',int((patent_summary.kept_figures>0).sum()))
print('结论：原始候选已通过文件与几何检查；下一 Notebook 将进行 DINO 视觉过滤和 embedding。')
print('Conclusion: Raw candidates passed file and geometry checks; the next notebook will apply DINO visual filtering and embedding.')


# ## 第6步：生成预览并保存总结 / Step 6: Generate a preview and save the summary

# In[ ]:


kept=figure_manifest[figure_manifest.keep_for_embedding].copy()
if len(kept):
    preview=kept.sample(min(40,len(kept)),random_state=42); columns=5; rows=math.ceil(len(preview)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=(15,3.2*rows)); axes=np.atleast_1d(axes).ravel()
    for ax,(_,item) in zip(axes,preview.iterrows()):
        ax.imshow(Image.open(item.figure_path).convert('RGB')); ax.set_title(f'{item.entry_id}\n{item.pdf} p{item.page} f{item.figure_number} conf={item.confidence:.2f}',fontsize=6); ax.axis('off')
    for ax in axes[len(preview):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(PREVIEW_ROOT/'swedish_yolo_kept_figures_preview.png',dpi=160,bbox_inches='tight'); plt.show()
summary={'created_at':datetime.now().isoformat(timespec='seconds'),'country':'SE','selected_drawing_pages':int(len(selected)),
    'matched_entries':int(selected.entry_id.nunique()),'source_pdf_files':int(len(selected.pdf_path.unique())),'render_dpi':RENDER_DPI,
    'yolo_model':str(MODEL_PATH.resolve()),'yolo_config_id':CONFIG_ID,'raw_figures':int(len(figure_manifest)),
    'geometry_kept_figures':int(figure_manifest.keep_for_embedding.sum()) if len(figure_manifest) else 0,'zero_detection_pages':int(len(zero_pages)),
    'fallback_detection_pages':int(fallback_pages),'segmentation_errors':int(len(segmentation_errors)),
    'entries_with_kept_figures':int((patent_summary.kept_figures>0).sum()),'figure_manifest':str((TABLE_ROOT/'swedish_yolo_figures_manifest.csv').resolve()),
    'patent_root':str(PATENT_ROOT.resolve())}
(OUTPUT_ROOT/'sweden_matched_segmentation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
display(pd.DataFrame([summary]).T.rename(columns={0:'value'}))
print('结论：配对瑞典图页的 YOLO 分割阶段完成，输出已按 entry_id 安全隔离。')
print('Conclusion: YOLO segmentation of matched Swedish drawing pages is complete, with outputs safely isolated by entry_id.')


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


# # Part 5 · 配对瑞典专利图 DINOv3 Embedding / DINOv3 Embeddings for Matched Swedish Patent Figures
# 
# 本 Notebook 对瑞典 YOLO 候选使用与芬兰端相同的 DINOv3 模型、宽松视觉过滤规则和归一化方式，然后分别写入每个配对条目的专利目录，并建立瑞典全局检索索引。这样两国向量可直接用余弦相似度比较。
# 
# This notebook applies the same DINOv3 model, permissive visual filter, and normalization used for Finnish YOLO candidates. It writes vectors into each matched-entry patent folder and builds a global Swedish retrieval index, making Swedish and Finnish vectors directly comparable with cosine similarity.

# ## 第0步：加载瑞典候选和模型 / Step 0: Load Swedish candidates and models

# In[ ]:


from pathlib import Path
from datetime import datetime
import json, math, time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import timm

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm").is_dir()), None)
assert ROOT is not None, "Could not find a parent directory containing embedding_atm."
if not (ROOT/'Swe-Fin-matched-patents').exists(): ROOT=ROOT.parent
SEGMENTATION_ROOT=ROOT/'embedding_atm'/'sweden_matched_segmentation'
INPUT_MANIFEST=SEGMENTATION_ROOT/'tables'/'swedish_yolo_figures_manifest.csv'
CLASSIFIER_PATH=ROOT/'embedding_atm'/'finnish_page_classifier_v1'/'models'/'finnish_page_classifier_dinov3_candidate_v1.joblib'
OUTPUT_ROOT=ROOT/'embedding_atm'/'sweden_figure_embedding'
TABLE_ROOT=OUTPUT_ROOT/'tables'; PREVIEW_ROOT=OUTPUT_ROOT/'previews'
INDEX_ROOT=ROOT/'embedding_atm'/'cross_country_retrieval'/'indexes'/'sweden'
RETRIEVAL_TABLE_ROOT=ROOT/'embedding_atm'/'cross_country_retrieval'/'tables'/'sweden'
for folder in [OUTPUT_ROOT,TABLE_ROOT,PREVIEW_ROOT,INDEX_ROOT,RETRIEVAL_TABLE_ROOT]: folder.mkdir(parents=True,exist_ok=True)
manifest=pd.read_csv(INPUT_MANIFEST); classifier_bundle=joblib.load(CLASSIFIER_PATH)
assert classifier_bundle['feature_variant']=='dino_only'
assert manifest.figure_path.map(lambda p:Path(p).exists()).all()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('YOLO crops:',len(manifest),'| Geometry kept:',int(manifest.keep_for_embedding.sum()),'| Device:',device)
print('结论：655 个瑞典 YOLO 候选和与芬兰端一致的 DINOv3 配置已加载。')
print('Conclusion: The 655 Swedish YOLO candidates and the Finnish-compatible DINOv3 configuration are loaded.')


# ## 第1步：生成或复用全部候选向量 / Step 1: Generate or reuse vectors for all candidates

# In[ ]:


MODEL_NAME=classifier_bundle['dino_model']; MODEL_SLUG='dinov3_vitb16'; INPUT_SIZE=256
model=timm.create_model(MODEL_NAME,pretrained=True,num_classes=0).to(device).eval(); embedding_dim=int(model.num_features)
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
def crop_tensor(path):
    image=Image.open(path).convert('RGB'); inner=int(INPUT_SIZE*0.92)
    contained=ImageOps.contain(image,(inner,inner),Image.Resampling.BICUBIC)
    canvas=Image.new('RGB',(INPUT_SIZE,INPUT_SIZE),'white'); canvas.paste(contained,((INPUT_SIZE-contained.width)//2,(INPUT_SIZE-contained.height)//2))
    array=np.asarray(canvas,dtype=np.float32)/255.0
    return (torch.from_numpy(array).permute(2,0,1)-mean)/std
embedding_path=OUTPUT_ROOT/'dinov3_yolo_crop_embeddings.npy'; key_path=TABLE_ROOT/'dinov3_yolo_crop_embedding_manifest.csv'
keys=[f'{Path(p).resolve()}|{Path(p).stat().st_size}' for p in manifest.figure_path]
cache_valid=False
if embedding_path.exists() and key_path.exists():
    cached=np.load(embedding_path); cached_keys=pd.read_csv(key_path).crop_key.tolist()
    cache_valid=cached.shape==(len(manifest),embedding_dim) and cached_keys==keys
if cache_valid:
    embeddings=cached.astype(np.float32); print('Reused cached Swedish crop embeddings.')
else:
    outputs=[]; batch_size=6 if device.type=='cpu' else 24; stage_start=time.time()
    with torch.inference_mode():
        for batch_start in range(0,len(manifest),batch_size):
            paths=manifest.figure_path.iloc[batch_start:batch_start+batch_size]
            batch=torch.stack([crop_tensor(path) for path in paths]).to(device)
            vectors=model(batch); vectors=vectors[0] if isinstance(vectors,(tuple,list)) else vectors
            outputs.append(F.normalize(vectors.float(),p=2,dim=1).cpu().numpy())
            if batch_start and batch_start%(batch_size*20)==0: print(f'Embedded {batch_start}/{len(manifest)} crops')
    embeddings=np.concatenate(outputs).astype(np.float32); np.save(embedding_path,embeddings)
    pd.DataFrame({'crop_index':np.arange(len(manifest)),'crop_key':keys}).to_csv(key_path,index=False,encoding='utf-8-sig')
    print(f'Embedding time: {(time.time()-stage_start)/60:.2f} minutes')
assert embeddings.shape==(len(manifest),embedding_dim)
assert np.allclose(np.linalg.norm(embeddings,axis=1),1.0,atol=1e-5)
print('结论：全部瑞典候选已转换为与芬兰端兼容的 768 维归一化 DINOv3 向量。')
print('Conclusion: Every Swedish candidate is converted into a Finnish-compatible normalized 768-dimensional DINOv3 vector.')


# ## 第2步：应用与芬兰端一致的视觉过滤 / Step 2: Apply the same visual filter used for Finland

# In[ ]:


visual_probability=classifier_bundle['model'].predict_proba(embeddings)[:,1]
CROP_DINO_MINIMUM=0.10; STAMP_MAXIMUM_PAGE_FRACTION=0.04; STAMP_MINIMUM_INK_RATIO=0.14
STAMP_MAXIMUM_DINO_PROBABILITY=0.99; STAMP_MINIMUM_COMPONENT_DENSITY_PER_10K=5.0
scored=manifest.copy(); scored['dino_drawing_probability']=visual_probability
scored['passes_dino_filter']=scored.dino_drawing_probability>=CROP_DINO_MINIMUM
scored['component_density_per_10k']=scored.connected_components_retained/(scored.width*scored.height)*10000
scored['stamp_like_review']=((scored.bbox_area_ratio<=STAMP_MAXIMUM_PAGE_FRACTION)&(scored.ink_ratio>=STAMP_MINIMUM_INK_RATIO)&
    (scored.dino_drawing_probability<STAMP_MAXIMUM_DINO_PROBABILITY)&(scored.component_density_per_10k>=STAMP_MINIMUM_COMPONENT_DENSITY_PER_10K))
scored['embedding_ready']=(scored.keep_for_embedding.astype(bool)&scored.passes_dino_filter)
scored['final_filter_reason']=np.where(~scored.keep_for_embedding.astype(bool),scored.quality_reason,
    np.where(~scored.passes_dino_filter,'dino_non_drawing','embedding_ready'))
scored['manual_filter_override']=''
scored.to_csv(TABLE_ROOT/'swedish_figures_embedding_ready_manifest.csv',index=False,encoding='utf-8-sig')
display(scored.groupby('final_filter_reason').size().rename('figures').to_frame())
print('DINO probability quantiles:',scored.dino_drawing_probability.quantile([0,0.1,0.25,0.5,0.75,0.9,1]).round(4).to_dict())
print('Stamp-like review candidates retained:',int(scored.stamp_like_review.sum()))
print('结论：瑞典 embedding_ready 由几何质量和 DINO 概率决定；小而密的图仅标记复核，不自动删除。')
print('Conclusion: Swedish embedding_ready uses geometry quality and DINO probability; small dense figures are review-flagged but not automatically removed.')


# ## 第3步：检查接受和拒绝样本 / Step 3: Inspect accepted and rejected samples

# In[ ]:


def save_preview(frame,path,title,limit=30):
    subset=frame.head(limit); columns=5; rows=max(1,math.ceil(len(subset)/columns))
    fig,axes=plt.subplots(rows,columns,figsize=(15,3.2*rows)); axes=np.atleast_1d(axes).ravel()
    for ax,(_,item) in zip(axes,subset.iterrows()):
        ax.imshow(Image.open(item.figure_path).convert('RGB')); ax.set_title(f'{item.entry_id}\n{item.pdf} p{item.page} f{item.figure_number}\nDINO={item.dino_drawing_probability:.3f}',fontsize=6); ax.axis('off')
    for ax in axes[len(subset):]: ax.axis('off')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=160,bbox_inches='tight'); plt.close(fig)
accepted=scored[scored.embedding_ready].sample(min(30,int(scored.embedding_ready.sum())),random_state=42)
rejected=scored[~scored.embedding_ready].sort_values(['final_filter_reason','dino_drawing_probability'])
save_preview(accepted,PREVIEW_ROOT/'accepted_swedish_figures.png','Accepted Swedish figures')
save_preview(rejected,PREVIEW_ROOT/'rejected_swedish_crops.png','Rejected Swedish crops')
display(Image.open(PREVIEW_ROOT/'rejected_swedish_crops.png'))
print('结论：接受与拒绝样本预览已生成，所有原始 YOLO 文件均保留。')
print('Conclusion: Accepted and rejected sample previews are generated, while every raw YOLO file remains preserved.')


# ## 第4步：写入每专利向量并建立瑞典索引 / Step 4: Save per-patent vectors and build the Swedish index

# In[ ]:


selected_positions=np.flatnonzero(scored.embedding_ready.to_numpy(dtype=bool)); selected=scored.iloc[selected_positions].reset_index(drop=True)
selected_embeddings=embeddings[selected_positions]; index_rows=[]; expected_by_dir={}
for global_index,(row,vector) in enumerate(zip(selected.itertuples(index=False),selected_embeddings)):
    pdf_stem=Path(row.pdf).stem; model_dir=SEGMENTATION_ROOT/'patents'/row.entry_id/pdf_stem/'embeddings'/MODEL_SLUG
    model_dir.mkdir(parents=True,exist_ok=True); vector_path=model_dir/(Path(row.figure_path).stem+'.npy')
    np.save(vector_path,vector.astype(np.float32)); expected_by_dir.setdefault(model_dir,set()).add(vector_path.name)
    index_rows.append({'sweden_figure_id':f'SE-FIG-{global_index+1:04d}','global_index':global_index,'entry_id':row.entry_id,'pdf':row.pdf,
        'page':int(row.page),'figure_number':int(row.figure_number),'figure_path':str(Path(row.figure_path).resolve()),
        'embedding_path':str(vector_path.resolve()),'model':MODEL_NAME,'embedding_dimension':embedding_dim,
        'yolo_confidence':float(row.confidence),'dino_filter_probability':float(row.dino_drawing_probability),'manual_filter_override':''})
for model_dir,expected in expected_by_dir.items():
    for stale in model_dir.glob('*.npy'):
        if stale.name not in expected: stale.unlink()
index_manifest=pd.DataFrame(index_rows)
matrix_path=INDEX_ROOT/f'{MODEL_SLUG}_embeddings.npy'; manifest_path=INDEX_ROOT/f'{MODEL_SLUG}_manifest.csv'
np.save(matrix_path,selected_embeddings.astype(np.float32)); index_manifest.to_csv(manifest_path,index=False,encoding='utf-8-sig')
np.savez_compressed(INDEX_ROOT/f'{MODEL_SLUG}_index_bundle.npz',embeddings=selected_embeddings.astype(np.float32),
    sweden_figure_ids=index_manifest.sweden_figure_id.to_numpy(dtype=str),entry_ids=index_manifest.entry_id.to_numpy(dtype=str),
    figure_paths=index_manifest.figure_path.to_numpy(dtype=str))
assert len(index_manifest)==len(selected_embeddings) and index_manifest.embedding_path.map(lambda p:Path(p).exists()).all()
print('Indexed figures:',len(index_manifest),'| Entries:',index_manifest.entry_id.nunique(),'| Shape:',selected_embeddings.shape)
print('结论：瑞典有效图的每专利向量、全局矩阵和索引清单已经生成。')
print('Conclusion: Per-patent vectors, the global matrix, and the index manifest are generated for valid Swedish figures.')


# ## 第5步：生成瑞典内部邻居表并保存总结 / Step 5: Build Swedish internal neighbours and save the summary

# In[ ]:


similarity=selected_embeddings@selected_embeddings.T; np.fill_diagonal(similarity,-np.inf)
top_k=5; neighbour_indices=np.argsort(-similarity,axis=1)[:,:top_k]; rows=[]
for query_index,neighbours in enumerate(neighbour_indices):
    query=index_manifest.iloc[query_index]
    for rank,neighbour_index in enumerate(neighbours,1):
        neighbour=index_manifest.iloc[int(neighbour_index)]
        rows.append({'query_figure_id':query.sweden_figure_id,'query_entry_id':query.entry_id,'query_pdf':query.pdf,'query_page':int(query.page),
            'query_figure_path':query.figure_path,'rank':rank,'neighbour_figure_id':neighbour.sweden_figure_id,'neighbour_entry_id':neighbour.entry_id,
            'neighbour_pdf':neighbour.pdf,'neighbour_page':int(neighbour.page),'neighbour_figure_path':neighbour.figure_path,
            'cosine_similarity':float(similarity[query_index,neighbour_index]),'same_entry':bool(query.entry_id==neighbour.entry_id)})
neighbour_table=pd.DataFrame(rows); neighbour_table.to_csv(RETRIEVAL_TABLE_ROOT/'sweden_top5_internal_neighbours.csv',index=False,encoding='utf-8-sig')
summary={'created_at':datetime.now().isoformat(timespec='seconds'),'country':'SE','raw_yolo_figures':int(len(scored)),
    'geometry_kept':int(scored.keep_for_embedding.sum()),'embedding_ready':int(scored.embedding_ready.sum()),
    'dino_rejected':int((scored.final_filter_reason=='dino_non_drawing').sum()),'stamp_like_review_retained':int(scored.stamp_like_review.sum()),
    'indexed_entries':int(index_manifest.entry_id.nunique()),'model':MODEL_NAME,'embedding_dimension':embedding_dim,
    'matrix_path':str(matrix_path.resolve()),'manifest_path':str(manifest_path.resolve()),
    'mean_top1_internal_similarity':float(neighbour_table[neighbour_table['rank']==1].cosine_similarity.mean()),'source_files_preserved':True}
(OUTPUT_ROOT/'sweden_figure_embedding_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(INDEX_ROOT/f'{MODEL_SLUG}_index_info.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
display(pd.DataFrame([summary]).T.rename(columns={0:'value'}))
print('结论：瑞典 embedding 和索引阶段完成，现可与芬兰图库进行跨国检索。')
print('Conclusion: Swedish embedding and indexing are complete and ready for cross-country retrieval against the Finnish gallery.')


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


# # Part 6 · 瑞典—芬兰跨国图像检索基线 / Swedish–Finnish Cross-Country Image Retrieval Baseline
# 
# 本 Notebook 不逐个打开 PDF 比较，而是一次性计算全部瑞典图与芬兰图的余弦相似度矩阵。图级结果再聚合为专利级候选，并利用已知 `entry_id` 配对计算 Recall@1、Recall@5、Recall@10 和 MRR。
# 
# This notebook does not compare PDFs one by one. It computes the full cosine-similarity matrix between Swedish and Finnish figure embeddings in one vectorized operation. Figure-level results are aggregated into patent candidates, and known `entry_id` pairs provide Recall@1, Recall@5, Recall@10, and MRR evaluation.

# ## 第0步：加载并验证两国索引 / Step 0: Load and validate both country indexes

# In[ ]:


from pathlib import Path
from datetime import datetime
import json, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm").is_dir()), None)
assert ROOT is not None, "Could not find a parent directory containing embedding_atm."
if not (ROOT/'Swe-Fin-matched-patents').exists(): ROOT=ROOT.parent
BASE=ROOT/'embedding_atm'/'cross_country_retrieval'; MODEL_SLUG='dinov3_vitb16'
FI_INDEX=BASE/'indexes'/'finland'; SE_INDEX=BASE/'indexes'/'sweden'
TABLE_ROOT=BASE/'tables'/'cross_country'; PREVIEW_ROOT=BASE/'previews'/'cross_country'
for folder in [TABLE_ROOT,PREVIEW_ROOT]: folder.mkdir(parents=True,exist_ok=True)
fi_manifest=pd.read_csv(FI_INDEX/f'{MODEL_SLUG}_manifest.csv'); se_manifest=pd.read_csv(SE_INDEX/f'{MODEL_SLUG}_manifest.csv')
fi_embeddings=np.load(FI_INDEX/f'{MODEL_SLUG}_embeddings.npy').astype(np.float32)
se_embeddings=np.load(SE_INDEX/f'{MODEL_SLUG}_embeddings.npy').astype(np.float32)
assert len(fi_manifest)==len(fi_embeddings) and len(se_manifest)==len(se_embeddings)
assert fi_embeddings.shape[1]==se_embeddings.shape[1]==768
assert np.allclose(np.linalg.norm(fi_embeddings,axis=1),1,atol=1e-5) and np.allclose(np.linalg.norm(se_embeddings,axis=1),1,atol=1e-5)
print('Swedish figures:',len(se_manifest),'| Finnish figures:',len(fi_manifest),'| Shared model:',se_manifest.model.iloc[0])
print('Swedish entries:',se_manifest.entry_id.nunique(),'| Finnish entries:',fi_manifest.entry_id.nunique())
print('结论：两国索引维度、归一化和模型一致，可以直接进行余弦相似度比较。')
print('Conclusion: Both country indexes share model, dimension, and normalization and are directly comparable with cosine similarity.')


# ## 第1步：向量化计算图级 Top-10 / Step 1: Compute vectorized figure-level Top-10 retrieval

# In[ ]:


similarity=se_embeddings@fi_embeddings.T
TOP_K=10; top_indices=np.argsort(-similarity,axis=1)[:,:TOP_K]; figure_rows=[]
for se_index,fi_indices in enumerate(top_indices):
    query=se_manifest.iloc[se_index]
    for rank,fi_index in enumerate(fi_indices,1):
        candidate=fi_manifest.iloc[int(fi_index)]
        figure_rows.append({'query_figure_id':query.sweden_figure_id,'query_entry_id':query.entry_id,'query_pdf':query.pdf,
            'query_page':int(query.page),'query_figure_path':query.figure_path,'rank':rank,'candidate_figure_id':candidate.finland_figure_id,
            'candidate_entry_id':candidate.entry_id,'candidate_pdf':candidate.pdf,'candidate_page':int(candidate.page),
            'candidate_figure_path':candidate.figure_path,'cosine_similarity':float(similarity[se_index,fi_index]),
            'is_known_pair':bool(query.entry_id==candidate.entry_id)})
figure_top10=pd.DataFrame(figure_rows)
figure_top10.to_csv(TABLE_ROOT/'sweden_to_finland_figure_top10.csv',index=False,encoding='utf-8-sig')
has_pair=se_manifest.entry_id.isin(set(fi_manifest.entry_id)); evaluable_figure_indices=np.flatnonzero(has_pair.to_numpy())
all_order=np.argsort(-similarity,axis=1); figure_true_ranks=[]
fi_entry_array=fi_manifest.entry_id.to_numpy()
for se_index in evaluable_figure_indices:
    target=se_manifest.iloc[se_index].entry_id
    figure_true_ranks.append(int(np.flatnonzero(fi_entry_array[all_order[se_index]]==target)[0]+1))
figure_true_ranks=np.asarray(figure_true_ranks)
print('Similarity matrix:',similarity.shape,'| comparisons:',similarity.size,'| evaluable Swedish figures:',len(figure_true_ranks))
print('Figure Recall@1/5/10:',*(round(float(np.mean(figure_true_ranks<=k)),4) for k in [1,5,10]))
print('结论：全部图像比较已由一次矩阵乘法完成，图级 Top-10 候选表已经生成。')
print('Conclusion: All figure comparisons are completed by one matrix multiplication and the figure-level Top-10 table is generated.')


# ## 第2步：聚合为专利级候选 / Step 2: Aggregate figure matches into patent-level candidates

# In[ ]:


se_entries=se_manifest.entry_id.drop_duplicates().tolist(); fi_entries=fi_manifest.entry_id.drop_duplicates().tolist()
se_entry_array=se_manifest.entry_id.to_numpy(); fi_entry_array=fi_manifest.entry_id.to_numpy()
se_groups={entry:np.flatnonzero(se_entry_array==entry) for entry in se_entries}
fi_groups={entry:np.flatnonzero(fi_entry_array==entry) for entry in fi_entries}
patent_rows=[]
for query_entry in se_entries:
    query_indices=se_groups[query_entry]
    for candidate_entry in fi_entries:
        candidate_indices=fi_groups[candidate_entry]; block=similarity[np.ix_(query_indices,candidate_indices)]
        local_position=np.unravel_index(int(np.argmax(block)),block.shape)
        best_se_index=int(query_indices[local_position[0]]); best_fi_index=int(candidate_indices[local_position[1]])
        query=se_manifest.iloc[best_se_index]; candidate=fi_manifest.iloc[best_fi_index]
        patent_rows.append({'query_entry_id':query_entry,'query_pdf':query.pdf,'candidate_entry_id':candidate_entry,'candidate_pdf':candidate.pdf,
            'max_figure_similarity':float(block[local_position]),'best_query_figure_id':query.sweden_figure_id,
            'best_query_figure_path':query.figure_path,'best_candidate_figure_id':candidate.finland_figure_id,
            'best_candidate_figure_path':candidate.figure_path,'is_known_pair':bool(query_entry==candidate_entry)})
patent_scores=pd.DataFrame(patent_rows)
patent_scores['rank']=patent_scores.groupby('query_entry_id').max_figure_similarity.rank(method='first',ascending=False).astype(int)
patent_scores=patent_scores.sort_values(['query_entry_id','rank']).reset_index(drop=True)
patent_scores.to_csv(TABLE_ROOT/'sweden_to_finland_all_patent_scores.csv',index=False,encoding='utf-8-sig')
patent_top10=patent_scores[patent_scores['rank']<=10].copy()
patent_top10.to_csv(TABLE_ROOT/'sweden_to_finland_patent_top10.csv',index=False,encoding='utf-8-sig')
print('Patent comparisons:',len(patent_scores),'| Top-10 rows:',len(patent_top10))
print('结论：每个瑞典条目均得到按最强图像证据排序的芬兰专利 Top-10 候选。')
print('Conclusion: Every Swedish entry now has a Finnish patent Top-10 ranked by its strongest figure-level evidence.')


# ## 第3步：使用已知配对评估检索 / Step 3: Evaluate retrieval with known patent pairs

# In[ ]:


true_pairs=patent_scores[patent_scores.is_known_pair].copy().sort_values('rank')
patent_true_ranks=true_pairs['rank'].to_numpy()
metrics={'evaluable_paired_entries':int(len(true_pairs)),'swedish_indexed_entries':int(len(se_entries)),
    'finnish_indexed_entries':int(len(fi_entries)),'median_true_pair_rank':float(np.median(patent_true_ranks)),
    'mean_reciprocal_rank':float(np.mean(1/patent_true_ranks)),'recall_at_1':float(np.mean(patent_true_ranks<=1)),
    'recall_at_3':float(np.mean(patent_true_ranks<=3)),'recall_at_5':float(np.mean(patent_true_ranks<=5)),
    'recall_at_10':float(np.mean(patent_true_ranks<=10)),'recall_at_20':float(np.mean(patent_true_ranks<=20)),
    'figure_recall_at_1':float(np.mean(figure_true_ranks<=1)),'figure_recall_at_5':float(np.mean(figure_true_ranks<=5)),
    'figure_recall_at_10':float(np.mean(figure_true_ranks<=10))}
pd.DataFrame([metrics]).to_csv(TABLE_ROOT/'cross_country_retrieval_metrics.csv',index=False,encoding='utf-8-sig')
true_pairs[['query_entry_id','query_pdf','candidate_pdf','max_figure_similarity','rank','best_query_figure_id','best_candidate_figure_id']].to_csv(
    TABLE_ROOT/'known_pair_patent_ranks.csv',index=False,encoding='utf-8-sig')
all_pair_entries=pd.read_csv(ROOT/'embedding_atm'/'finnish_page_classifier_pilot'/'tables'/'matched_patent_pairs.csv')[['entry_id','pair_complete']]
coverage=all_pair_entries.copy(); coverage['has_sweden_figures']=coverage.entry_id.isin(set(se_entries)); coverage['has_finland_figures']=coverage.entry_id.isin(set(fi_entries))
coverage['retrieval_evaluable']=coverage.has_sweden_figures&coverage.has_finland_figures
coverage['coverage_status']=np.select([coverage.retrieval_evaluable,coverage.has_sweden_figures,coverage.has_finland_figures],
    ['both_indexes','sweden_only','finland_only'],default='neither_index')
coverage.to_csv(TABLE_ROOT/'matched_entry_index_coverage.csv',index=False,encoding='utf-8-sig')
display(pd.DataFrame([metrics]).T.rename(columns={0:'value'}))
display(coverage.coverage_status.value_counts().rename('entries').to_frame())
print('结论：89 个具有两国图像的已知配对可评估；其专利级 Recall@1、@5、@10 已量化。')
print('Conclusion: 89 known pairs with figures in both indexes are evaluable, with patent-level Recall@1, @5, and @10 quantified.')


# ## 第4步：生成正确与错误候选的可视化 / Step 4: Visualize correct and incorrect patent candidates

# In[ ]:


evaluation_entries=true_pairs.sort_values('query_entry_id').query_entry_id.tolist()
chosen_positions=np.linspace(0,len(evaluation_entries)-1,12,dtype=int); chosen=[evaluation_entries[i] for i in chosen_positions]
fig,axes=plt.subplots(len(chosen),3,figsize=(12,3.2*len(chosen)))
for row_number,entry in enumerate(chosen):
    predicted=patent_scores[(patent_scores.query_entry_id==entry)&(patent_scores['rank']==1)].iloc[0]
    truth=patent_scores[(patent_scores.query_entry_id==entry)&(patent_scores.is_known_pair)].iloc[0]
    axes[row_number,0].imshow(Image.open(predicted.best_query_figure_path).convert('RGB')); axes[row_number,0].set_title(f'SE query: {entry}\n{predicted.query_pdf}',fontsize=7)
    axes[row_number,1].imshow(Image.open(predicted.best_candidate_figure_path).convert('RGB')); axes[row_number,1].set_title(f'Predicted rank 1: {predicted.candidate_entry_id}\ncos={predicted.max_figure_similarity:.3f} pair={predicted.is_known_pair}',fontsize=7)
    axes[row_number,2].imshow(Image.open(truth.best_candidate_figure_path).convert('RGB')); axes[row_number,2].set_title(f'True FI pair: rank {truth["rank"]}\ncos={truth.max_figure_similarity:.3f}',fontsize=7)
    for column in range(3): axes[row_number,column].axis('off')
fig.tight_layout(); preview_path=PREVIEW_ROOT/'patent_retrieval_examples.png'
fig.savefig(preview_path,dpi=150,bbox_inches='tight'); plt.show()
print('结论：12 个专利查询的预测第一名与真实芬兰配对对照图已经生成。')
print('Conclusion: A 12-query comparison of predicted rank-1 candidates and true Finnish pairs is generated.')


# ## 第5步：保存跨国基线总结 / Step 5: Save the cross-country baseline summary

# In[ ]:


summary={'created_at':datetime.now().isoformat(timespec='seconds'),'experiment':'sweden_to_finland_dinov3_baseline',
    'model':se_manifest.model.iloc[0],'embedding_dimension':int(se_embeddings.shape[1]),'swedish_figures':int(len(se_manifest)),
    'finnish_figures':int(len(fi_manifest)),'figure_comparisons':int(similarity.size),'patent_score':'maximum_cross_figure_cosine',**metrics,
    'figure_top10_table':str((TABLE_ROOT/'sweden_to_finland_figure_top10.csv').resolve()),
    'patent_top10_table':str((TABLE_ROOT/'sweden_to_finland_patent_top10.csv').resolve()),
    'coverage_table':str((TABLE_ROOT/'matched_entry_index_coverage.csv').resolve()),'preview':str(preview_path.resolve())}
(BASE/'cross_country_retrieval_baseline_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
display(pd.DataFrame([summary]).T.rename(columns={0:'value'}))
print('结论：首个瑞典到芬兰的跨国检索基线已经完成；结果是实验基线，不是最终模型结论。')
print('Conclusion: The first Swedish-to-Finnish retrieval baseline is complete; these results are an experimental baseline, not a final model claim.')


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


# # Part 7 · 专利级多图汇总实验 / Patent-level multi-figure pooling experiment
# 
# 本 notebook 保存第一轮人工复核意见，并在不重新分割、不重新计算 embedding 的情况下，比较多种瑞典–芬兰专利级图片相似度汇总方法。
# 
# This notebook records the first human review round and compares patent-level Swedish–Finnish image-similarity pooling methods without rerunning segmentation or embeddings.
# 
# 所有表格字段和标签使用英文；说明和每一步结论使用中英双语。选择 Run All 可完整复现结果。
# 
# All table fields and labels are English-only; explanations and step conclusions are bilingual. Choose Run All to reproduce the experiment.

# ## 1 · 载入索引与固定输入 / Load indexes and freeze inputs
# 
# 读取现有 DINOv3 embedding、瑞典与芬兰图片清单，以及已知专利配对。这个实验只改变专利分数的汇总方式。
# 
# Load the existing DINOv3 embeddings, figure manifests, and known patent pairs. Only patent-level pooling changes in this experiment.

# In[ ]:


from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm" / "cross_country_retrieval").is_dir()), None)
assert ROOT is not None, "Open this notebook from the materials workspace."
BASE = ROOT / "embedding_atm" / "cross_country_retrieval"
MODEL = "dinov3_vitb16"
OUT = BASE / "tables" / "pooling_experiment"
OUT.mkdir(parents=True, exist_ok=True)
SE_MANIFEST_PATH = BASE / "indexes" / "sweden" / f"{MODEL}_manifest.csv"
FI_MANIFEST_PATH = BASE / "indexes" / "finland" / f"{MODEL}_manifest.csv"
SE_EMBED_PATH = BASE / "indexes" / "sweden" / f"{MODEL}_embeddings.npy"
FI_EMBED_PATH = BASE / "indexes" / "finland" / f"{MODEL}_embeddings.npy"
REFERENCE_DIR = next((p for p in [
    ROOT / "embedding_atm" / "github" / "reference_annotations",
    *[p / "reference_annotations" for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]],
] if p.is_dir()), None)
assert REFERENCE_DIR is not None, "Bundled reference_annotations directory was not found."
ANNOTATION_PATH = REFERENCE_DIR / "figure_pair_ai_annotations.csv"
sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

se_manifest = pd.read_csv(SE_MANIFEST_PATH)
fi_manifest = pd.read_csv(FI_MANIFEST_PATH)
se_embeddings = np.load(SE_EMBED_PATH).astype(np.float32)
fi_embeddings = np.load(FI_EMBED_PATH).astype(np.float32)
assert len(se_manifest) == len(se_embeddings)
assert len(fi_manifest) == len(fi_embeddings)
assert np.allclose(np.linalg.norm(se_embeddings, axis=1), 1, atol=.02)
assert np.allclose(np.linalg.norm(fi_embeddings, axis=1), 1, atol=.02)

se_entries = se_manifest.entry_id.drop_duplicates().tolist()
fi_entries = fi_manifest.entry_id.drop_duplicates().tolist()
paired_entries = sorted(set(se_entries) & set(fi_entries))
input_hashes = {
    "sweden_manifest": sha256(SE_MANIFEST_PATH), "finland_manifest": sha256(FI_MANIFEST_PATH),
    "sweden_embeddings": sha256(SE_EMBED_PATH), "finland_embeddings": sha256(FI_EMBED_PATH),
}
print(f"Swedish figures: {len(se_manifest)} | Finnish figures: {len(fi_manifest)}")
print(f"Swedish patents: {len(se_entries)} | Finnish patents: {len(fi_entries)} | Known evaluable pairs: {len(paired_entries)}")
print("结论：固定的 DINOv3 索引已载入；本实验只比较专利级分数汇总。")
print("Conclusion: The fixed DINOv3 indexes are loaded; this experiment changes patent-level pooling only.")


# ## 2 · 保存人工复核反馈 / Record human review feedback
# 
# human_relation 表示图像关系；match_scope 单独记录完整图、局部/组合图、不同视图或分割失败。这样不会把“属于同一机构”和“视觉上是同一幅图”混为一谈。
# 
# human_relation records figure relation, while match_scope separately records full matches, partial/composite matches, different views, and segmentation failures. This keeps invention identity separate from visual figure identity.
# 
# 没有被用户明确提到的编号不会自动改成负样本。反馈写入独立表，不覆盖 AI 标注。
# 
# Unmentioned rows are not converted to negatives. Feedback is exported separately and does not overwrite AI annotations.

# In[ ]:


HUMAN_REVIEW = [
("ANN-0003","related_figure","multi_crop_composition","Together with ANN-0004, the Swedish crops compose the Finnish figure."),
("ANN-0004","same_figure","partial_composite_match","The Swedish crop is one part of the Finnish composite; ANN-0003 supplies another part."),
("ANN-0005","different_figure","missing_corresponding_view","The displayed figures differ; Finnish Fig. 6 was not found in the current Swedish figures."),
("ANN-0006","different_figure","full_view","Not the same displayed figure."),
("ANN-0007","same_figure","full_view","Same figure."),
("ANN-0009","different_figure","full_view","Not the same displayed figure."),
("ANN-0010","different_figure","full_view","Not the same displayed figure."),
("ANN-0013","same_figure","candidate_partial","Same figure, but part of the Finnish crop is missing."),
("ANN-0015","different_figure","full_view","Clearly different figures."),
("ANN-0018","different_figure","full_view","Different figures."),
("ANN-0019","different_figure","full_view","Different figures."),
("ANN-0020","different_figure","full_view","Different figures."),
("ANN-0024","different_figure","query_split","The Swedish segmentation did not combine the two relevant parts; the displayed pair is not the same figure."),
("ANN-0025","different_figure","full_view","Different figures."),
("ANN-0028","uncertain","insufficient_visual_evidence","No consistent image could be established."),
("ANN-0029","same_figure","full_view","Same figure."),
("ANN-0030","different_figure","full_view","Different figures."),
("ANN-0033","related_figure","different_view_same_mechanism","Prior information says the mechanism is the same, but the displayed figures are not the same view."),
("ANN-0043","same_figure","full_view","Same figure."),
("ANN-0046","same_figure","partial_composite_match","The matching Finnish part is accompanied by other figures."),
("ANN-0051","same_figure","partial_composite_match","Same figure with extra content caused by segmentation."),
("ANN-0053","same_figure","partial_composite_match","Same figure with extra content caused by segmentation."),
("ANN-0071","same_figure","full_view","Same figure."),
("ANN-0076","related_figure","different_view_same_mechanism","Same mechanism, visually different view."),
("ANN-0077","related_figure","different_view_same_mechanism","Same mechanism, visually different view."),
("ANN-0082","same_figure","partial_composite_match","Same figure; segmentation changed the visible extent."),
("ANN-0087","same_figure","partial_composite_match","Same figure, with other figures included in the Finnish crop."),
("ANN-0091","same_figure","partial_composite_match","Same figure, with extra content in the candidate crop."),
("ANN-0101","same_figure","partial_composite_match","Same figure; segmentation included additional content."),
("ANN-0105","same_figure","query_fragmented","Same figure."),
("ANN-0106","same_figure","query_fragmented","The Swedish patent figure was not successfully assembled, but the parts represent the same object."),
("ANN-0110","related_figure","segmentation_failure","Same mechanism, but segmentation failure prevents a reliable same-view judgment."),
("ANN-0111","related_figure","segmentation_failure","Same mechanism, but segmentation failure prevents a reliable same-view judgment."),
]
feedback = pd.DataFrame(HUMAN_REVIEW, columns=["annotation_id","human_relation","match_scope","human_notes"])
assert feedback.annotation_id.is_unique
assert set(feedback.human_relation) <= {"same_figure","related_figure","different_figure","uncertain","bad_crop"}
annotations = pd.read_csv(ANNOTATION_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
feedback = feedback.merge(
    annotations[["annotation_id","case_id","query_entry_id","candidate_entry_id","query_figure_id",
                 "candidate_figure_id","ai_relation","ai_reason"]],
    on="annotation_id", how="left", validate="one_to_one"
)
assert feedback.case_id.notna().all()
feedback["review_source"] = "user_review_round_1"
feedback["review_date"] = "2026-11-08"
feedback["ai_human_agreement"] = feedback.human_relation.eq(feedback.ai_relation)
feedback.to_csv(OUT / "human_review_feedback_round1.csv", index=False, encoding="utf-8-sig")
display(feedback.human_relation.value_counts().rename_axis("human_relation").reset_index(name="count"))
print(f"Recorded rows: {len(feedback)} | AI-human exact agreement: {feedback.ai_human_agreement.mean():.1%}")
print("结论：两轮明确反馈已独立保存；未提及样本保持未复核状态。")
print("Conclusion: Both explicit feedback rounds are recorded separately; unmentioned pairs remain unreviewed.")


# ## 3 · 建立图片相似度矩阵 / Build the figure-similarity matrix
# 
# 计算全部瑞典–芬兰图片余弦相似度。每个专利对形成一个小矩阵：行是瑞典图片，列是芬兰图片。
# 
# Compute all Swedish–Finnish cosine similarities. Each patent pair becomes a small matrix whose rows are Swedish figures and columns are Finnish figures.

# In[ ]:


similarity = se_embeddings @ fi_embeddings.T
se_array = se_manifest.entry_id.to_numpy()
fi_array = fi_manifest.entry_id.to_numpy()
se_groups = {entry: np.flatnonzero(se_array == entry) for entry in se_entries}
fi_groups = {entry: np.flatnonzero(fi_array == entry) for entry in fi_entries}
print("Similarity matrix:", similarity.shape, "| comparisons:", similarity.size)
print("结论：所有图对相似度已一次性计算，可用不同规则汇总而无需重新 embedding。")
print("Conclusion: All figure-pair similarities are computed once and can be pooled without re-embedding.")


# ## 4 · 定义专利级汇总方法 / Define patent-level pooling
# 
# 比较以下不需要训练的规则：
# 
# - max_pair：最高单个图对分数，即当前 baseline。
# - all_pair_mean：全部图对直接平均，用作对照。
# - query_best_mean：每张瑞典图先找芬兰专利中的最佳匹配，再平均。
# - symmetric_best_mean：同时考虑每张瑞典图和每张芬兰图的最佳对应。
# - greedy_top2、greedy_top3：选择最高的 2 或 3 个不重复图对后平均，避免一张图重复提供多份证据。
# - top3_any_pair：最高 3 个图对直接平均，展示不去重可能产生的影响。
# 
# Compare training-free max, all-pair mean, directed best-match mean, symmetric best-match mean, and top matching-pair averages with and without one-to-one deduplication.

# In[ ]:


def greedy_unique_mean(block, requested):
    order = np.argsort(-block, axis=None)
    used_rows, used_cols, values = set(), set(), []
    target = min(requested, min(block.shape))
    for flat in order:
        row, col = np.unravel_index(int(flat), block.shape)
        if row not in used_rows and col not in used_cols:
            values.append(float(block[row, col]))
            used_rows.add(row); used_cols.add(col)
            if len(values) == target:
                break
    return float(np.mean(values))

def pooling_scores(block):
    flat = np.sort(block.ravel())[::-1]
    return {
        "max_pair": float(flat[0]),
        "all_pair_mean": float(block.mean()),
        "query_best_mean": float(block.max(axis=1).mean()),
        "symmetric_best_mean": float((block.max(axis=1).mean() + block.max(axis=0).mean()) / 2),
        "greedy_top2": greedy_unique_mean(block, 2),
        "greedy_top3": greedy_unique_mean(block, 3),
        "top3_any_pair": float(flat[:min(3, len(flat))].mean()),
    }

rows = []
for query_entry in se_entries:
    query_indices = se_groups[query_entry]
    for candidate_entry in fi_entries:
        candidate_indices = fi_groups[candidate_entry]
        block = similarity[np.ix_(query_indices, candidate_indices)]
        row = {"query_entry_id": query_entry, "candidate_entry_id": candidate_entry,
               "is_known_pair": query_entry == candidate_entry,
               "query_figure_count": len(query_indices), "candidate_figure_count": len(candidate_indices)}
        row.update(pooling_scores(block))
        rows.append(row)
scores = pd.DataFrame(rows)
methods = ["max_pair","all_pair_mean","query_best_mean","symmetric_best_mean",
           "greedy_top2","greedy_top3","top3_any_pair"]
assert len(scores) == len(se_entries) * len(fi_entries)
print("Patent comparisons:", len(scores), "| Methods:", len(methods))
print("结论：七种专利级汇总分数已生成，其中两种专门保留多张不重复的匹配证据。")
print("Conclusion: Seven patent-level scores are generated, including two that retain multiple unique matching pairs.")


# ## 5 · 比较正确专利排名 / Compare known-pair ranks
# 
# 对 89 个同时具有两国图片的已知配对，计算 MRR、正确专利中位排名及 Recall@1/3/5/10/20。这里的已知配对用于探索方法，不是独立测试集，因此不能把最高结果当成最终泛化性能。
# 
# For 89 known pairs with figures in both countries, compute MRR, median rank, and Recall@1/3/5/10/20. These known pairs support method exploration and are not an independent test set.

# In[ ]:


rank_frames = []
metric_rows = []
for method in methods:
    ranked = scores[["query_entry_id","candidate_entry_id","is_known_pair",method]].copy()
    ranked["rank"] = ranked.groupby("query_entry_id")[method].rank(method="first", ascending=False).astype(int)
    truth = ranked[ranked.is_known_pair].copy()
    truth["method"] = method
    truth = truth.rename(columns={method:"true_pair_score"})
    rank_frames.append(truth[["query_entry_id","method","true_pair_score","rank"]])
    ranks = truth["rank"].to_numpy()
    metric_rows.append({
        "method": method, "evaluable_pairs": len(ranks), "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1 / ranks)),
        **{f"recall_at_{k}": float(np.mean(ranks <= k)) for k in [1,3,5,10,20]}
    })
true_ranks = pd.concat(rank_frames, ignore_index=True)
metrics = pd.DataFrame(metric_rows).sort_values(
    ["recall_at_1","recall_at_5","mrr"], ascending=False
).reset_index(drop=True)
display(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c not in ["method","evaluable_pairs"]}))
print("Best exploratory method by Recall@1, Recall@5, then MRR:", metrics.iloc[0].method)
print("结论：表中直接显示多图汇总是否超过当前 max-pair baseline；方法选择仍需独立专利集验证。")
print("Conclusion: The table shows whether multi-figure pooling improves on max-pair; final selection still requires an independent patent set.")


# ## 6 · 配对 bootstrap 与困难案例变化 / Paired bootstrap and hard-case changes
# 
# 用相同的 89 个查询做配对 bootstrap，估计各方法相对 max_pair 的 Recall@1 差异范围；并单独查看原 23 个困难案例的排名变化。
# 
# Use paired bootstrap over the same 89 queries to estimate Recall@1 changes versus max_pair, and inspect rank changes for the original 23 hard cases.

# In[ ]:


wide = true_ranks.pivot(index="query_entry_id", columns="method", values="rank").loc[paired_entries]
rng = np.random.default_rng(20261108)
bootstrap_rows = []
baseline_hit = (wide["max_pair"].to_numpy() <= 1).astype(float)
for method in methods:
    method_hit = (wide[method].to_numpy() <= 1).astype(float)
    differences = []
    for _ in range(5000):
        sample = rng.integers(0, len(wide), len(wide))
        differences.append(float((method_hit[sample] - baseline_hit[sample]).mean()))
    bootstrap_rows.append({
        "method": method,
        "recall_at_1_difference_vs_max": float((method_hit - baseline_hit).mean()),
        "bootstrap_95_low": float(np.quantile(differences, .025)),
        "bootstrap_95_high": float(np.quantile(differences, .975)),
    })
bootstrap = pd.DataFrame(bootstrap_rows).sort_values("recall_at_1_difference_vs_max", ascending=False)
display(bootstrap.style.format({c:"{:+.3f}" for c in bootstrap.columns if c != "method"}))

case_summary = pd.read_csv(REFERENCE_DIR / "hard_case_summary.csv")
hard_entries = case_summary.query_entry_id.drop_duplicates().tolist()
hard = wide.loc[wide.index.intersection(hard_entries)].copy()
best_method = metrics.iloc[0].method
hard["rank_change_vs_max"] = hard["max_pair"] - hard[best_method]
hard = hard.reset_index().sort_values("rank_change_vs_max", ascending=False)
display(hard[["query_entry_id","max_pair",best_method,"rank_change_vs_max"]])
print("结论：bootstrap 显示改进的不确定范围；困难案例表显示哪些专利因多图证据上升或下降。")
print("Conclusion: Bootstrap shows uncertainty around improvements; the hard-case table shows which patents rise or fall with multi-figure evidence.")


# ## 7 · 导出结果与下一步 / Export results and next step
# 
# 导出所有专利分数、指标、正确配对排名、bootstrap 和困难案例变化。下一步根据本结果决定是否保留某种多图汇总，再增加原始绘图页或较大区域 embedding，作为分割失败的补充通道。
# 
# Export all patent scores, metrics, known-pair ranks, bootstrap results, and hard-case changes. Next, retain a justified pooling method and add page or large-region embeddings as a fallback for segmentation failures.

# In[ ]:


scores.to_csv(OUT / "all_patent_pooling_scores.csv", index=False, encoding="utf-8-sig")
metrics.to_csv(OUT / "pooling_metrics.csv", index=False, encoding="utf-8-sig")
true_ranks.to_csv(OUT / "known_pair_ranks_by_pooling.csv", index=False, encoding="utf-8-sig")
bootstrap.to_csv(OUT / "recall1_bootstrap_vs_max.csv", index=False, encoding="utf-8-sig")
hard.to_csv(OUT / "hard_case_rank_changes.csv", index=False, encoding="utf-8-sig")
summary = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model": MODEL, "swedish_figures": len(se_manifest), "finnish_figures": len(fi_manifest),
    "evaluable_known_pairs": len(paired_entries), "methods": methods,
    "best_exploratory_method": best_method,
    "best_metrics": metrics.iloc[0].to_dict(),
    "human_feedback_rows": len(feedback),
    "input_sha256": input_hashes,
    "limitations": [
        "Known pairs are used for exploratory method comparison, not independent final validation.",
        "Pooling cannot recover visual content that segmentation removed.",
        "Partial and composite matches motivate a future page or large-region embedding fallback."
    ]
}
(OUT / "pooling_experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
assert all((OUT / name).is_file() for name in [
    "human_review_feedback_round1.csv","all_patent_pooling_scores.csv","pooling_metrics.csv",
    "known_pair_ranks_by_pooling.csv","recall1_bootstrap_vs_max.csv","hard_case_rank_changes.csv",
    "pooling_experiment_summary.json"])
display(metrics)
print("结论：多图汇总实验及人工反馈均已导出；下一步不需要拼图即可先验证专利级多证据检索。")
print("Conclusion: Multi-figure pooling results and human feedback are exported; patent-level multi-evidence retrieval can be evaluated before image assembly.")


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


# # Part 8 · 原页区域补充检索 / Drawing-page region fallback retrieval
# 
# 本 notebook 为分割不完整、多个图被合并、或一幅图被拆开的情况增加第二条检索通道：对每张绘图页计算“整页 + 四个重叠大区域”的 DINOv3 embedding，再与 notebook 25 的裁剪图专利分数融合。
# 
# This notebook adds a second retrieval channel for incomplete, merged, or fragmented segmentation. It embeds each drawing page as one full page plus four overlapping large regions, then fuses page-region evidence with the figure-crop patent scores from notebook 25.
# 
# 表格字段为英文；说明及每一步结论中英双语。第一次 Run All 会计算并缓存原页区域 embedding，之后会复用缓存。
# 
# Table fields are English-only; explanations and step conclusions are bilingual. The first Run All computes and caches page-region embeddings; later runs reuse the cache.

# ## 1 · 读取绘图页和裁剪图基线 / Load drawing pages and the crop baseline
# 
# 读取已保存的 214 张瑞典绘图页、151 张芬兰绘图页，以及 notebook 25 的裁剪图汇总分数。无需重新下载或重新分割 PDF。
# 
# Load 214 Swedish and 151 Finnish drawing pages plus the crop-based pooling scores from notebook 25. No PDF download or segmentation rerun is required.

# In[ ]:


from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from IPython.display import display
import torch
import torch.nn.functional as F
import timm

ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "embedding_atm" / "cross_country_retrieval").is_dir()), None)
assert ROOT is not None, "Open this notebook from the materials workspace."
BASE = ROOT / "embedding_atm" / "cross_country_retrieval"
MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
MODEL_SLUG = "dinov3_vitb16"
SWEDEN_PAGES = ROOT / "embedding_atm/sweden_matched_segmentation/tables/rendered_swedish_drawing_pages.csv"
FINLAND_PAGES = ROOT / "embedding_atm/finland_segmentation/tables/rendered_finnish_drawing_pages.csv"
POOLING_SCORES = BASE / "tables/pooling_experiment/all_patent_pooling_scores.csv"
FEEDBACK_PATH = BASE / "tables/pooling_experiment/human_review_feedback_round1.csv"
OUT = BASE / "tables/page_region_fallback"
CACHE = BASE / "indexes/page_regions"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

se_pages = pd.read_csv(SWEDEN_PAGES)
fi_pages = pd.read_csv(FINLAND_PAGES)
figure_scores = pd.read_csv(POOLING_SCORES)
feedback = pd.read_csv(FEEDBACK_PATH)
for frame in [se_pages, fi_pages]:
    assert frame.rendered_page_path.map(lambda p: Path(p).is_file()).all()
assert {"symmetric_best_mean","max_pair"} <= set(figure_scores.columns)
paired_entries = sorted(set(figure_scores.loc[figure_scores.is_known_pair, 'query_entry_id']) & set(se_pages.entry_id) & set(fi_pages.entry_id))
print(f"Swedish pages: {len(se_pages)} | Finnish pages: {len(fi_pages)} | Evaluable known pairs: {len(paired_entries)}")
print("结论：原始绘图页和裁剪图基线均已载入，不需要重新分割。")
print("Conclusion: Drawing pages and the crop baseline are loaded; no segmentation rerun is needed.")


# ## 2 · 定义大区域并建立缓存键 / Define large regions and cache keys
# 
# 每页保留整页，再取四个带 10% 中央重叠的象限。这样既保留整体布局，也让较小的局部结构在 256×256 输入中更清楚。
# 
# Retain the full page and four quadrants with 10% overlap around the center. This preserves layout while making smaller structures more visible at 256×256 input size.

# In[ ]:


REGION_NAMES = ["full","top_left","top_right","bottom_left","bottom_right"]
def region_box(width, height, name):
    overlap_x, overlap_y = int(width * .05), int(height * .05)
    mid_x, mid_y = width // 2, height // 2
    boxes = {
        "full": (0, 0, width, height),
        "top_left": (0, 0, min(width, mid_x + overlap_x), min(height, mid_y + overlap_y)),
        "top_right": (max(0, mid_x - overlap_x), 0, width, min(height, mid_y + overlap_y)),
        "bottom_left": (0, max(0, mid_y - overlap_y), min(width, mid_x + overlap_x), height),
        "bottom_right": (max(0, mid_x - overlap_x), max(0, mid_y - overlap_y), width, height),
    }
    return boxes[name]

def make_region_manifest(frame, country):
    records = []
    for page in frame.itertuples(index=False):
        path = Path(page.rendered_page_path).resolve()
        stat = path.stat()
        with Image.open(path) as image:
            width, height = image.size
        for name in REGION_NAMES:
            records.append({
                "country": country, "entry_id": page.entry_id, "pdf": page.pdf, "page": int(page.page),
                "region_name": name, "page_path": str(path), "region_box": ",".join(map(str, region_box(width,height,name))),
                "cache_key": f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{name}|overlap10",
            })
    return pd.DataFrame(records)

se_regions = make_region_manifest(se_pages, "sweden")
fi_regions = make_region_manifest(fi_pages, "finland")
print(f"Swedish regions: {len(se_regions)} | Finnish regions: {len(fi_regions)}")
print("结论：每页生成五个互补视图；区域只在内存中裁取，不修改原始页图。")
print("Conclusion: Five complementary views are defined per page; regions are cropped in memory without changing page images.")


# ## 3 · 计算或复用 DINOv3 原页区域 embedding / Compute or reuse page-region embeddings
# 
# 使用与裁剪图完全相同的 DINOv3 模型、白底等比例缩放和归一化。缓存键包含原图路径、大小、修改时间和区域名称；输入改变时缓存会自动失效。
# 
# Use the same DINOv3 model, white-background contain resize, and normalization as the crop pipeline. Cache keys include path, size, modification time, and region name, so changed inputs invalidate the cache.

# In[ ]:


INPUT_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

def region_tensor(row):
    image = Image.open(row.page_path).convert("RGB")
    image = image.crop(tuple(map(int, row.region_box.split(","))))
    inner = int(INPUT_SIZE * .92)
    contained = ImageOps.contain(image, (inner,inner), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (INPUT_SIZE,INPUT_SIZE), "white")
    canvas.paste(contained, ((INPUT_SIZE-contained.width)//2, (INPUT_SIZE-contained.height)//2))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2,0,1) - mean) / std

def embed_regions(manifest, country):
    embedding_path = CACHE / f"{country}_{MODEL_SLUG}_page_region_embeddings.npy"
    manifest_path = CACHE / f"{country}_{MODEL_SLUG}_page_region_manifest.csv"
    cache_valid = False
    if embedding_path.exists() and manifest_path.exists():
        cached_manifest = pd.read_csv(manifest_path)
        cached = np.load(embedding_path)
        cache_valid = (
            cached.shape == (len(manifest), 768)
            and cached_manifest.cache_key.tolist() == manifest.cache_key.tolist()
        )
    if cache_valid:
        print(f"Reused cached {country} page-region embeddings.")
        return cached.astype(np.float32)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device).eval()
    outputs = []
    batch_size = 6 if device.type == "cpu" else 32
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(manifest), batch_size):
            batch = torch.stack([region_tensor(row) for row in manifest.iloc[start:start+batch_size].itertuples(index=False)]).to(device)
            vectors = model(batch)
            vectors = vectors[0] if isinstance(vectors, (tuple,list)) else vectors
            outputs.append(F.normalize(vectors.float(), p=2, dim=1).cpu().numpy())
            if start and start % (batch_size * 20) == 0:
                print(f"{country}: {start}/{len(manifest)} regions")
    embeddings = np.concatenate(outputs).astype(np.float32)
    np.save(embedding_path, embeddings)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"Embedded {country} regions in {(time.time()-started)/60:.2f} minutes.")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings

se_page_embeddings = embed_regions(se_regions, "sweden")
fi_page_embeddings = embed_regions(fi_regions, "finland")
assert se_page_embeddings.shape == (len(se_regions),768)
assert fi_page_embeddings.shape == (len(fi_regions),768)
assert np.allclose(np.linalg.norm(se_page_embeddings,axis=1),1,atol=1e-5)
assert np.allclose(np.linalg.norm(fi_page_embeddings,axis=1),1,atol=1e-5)
print("Device:", device)
print("结论：所有原页大区域已转换为与裁剪图兼容的归一化 DINOv3 embedding。")
print("Conclusion: All page regions are converted into normalized DINOv3 embeddings compatible with the crop index.")


# ## 4 · 计算原页专利分数 / Compute page-region patent scores
# 
# 对每个专利对计算区域相似度矩阵，并使用 notebook 25 中表现最好的双向逐区域最佳匹配平均。它允许一方保留组合图、另一方只包含局部图。
# 
# For each patent pair, compute a region-similarity block and apply symmetric directed-best pooling, the strongest exploratory method from notebook 25. It supports composite content on one side and partial content on the other.

# In[ ]:


page_similarity = se_page_embeddings @ fi_page_embeddings.T
se_array = se_regions.entry_id.to_numpy()
fi_array = fi_regions.entry_id.to_numpy()
se_groups = {entry: np.flatnonzero(se_array == entry) for entry in se_regions.entry_id.unique()}
fi_groups = {entry: np.flatnonzero(fi_array == entry) for entry in fi_regions.entry_id.unique()}
page_rows = []
for query_entry, se_index in se_groups.items():
    for candidate_entry, fi_index in fi_groups.items():
        block = page_similarity[np.ix_(se_index, fi_index)]
        page_rows.append({
            "query_entry_id": query_entry, "candidate_entry_id": candidate_entry,
            "page_region_symmetric_best": float((block.max(axis=1).mean() + block.max(axis=0).mean()) / 2),
            "page_region_max": float(block.max()),
        })
page_scores = pd.DataFrame(page_rows)
combined = figure_scores.merge(page_scores, on=["query_entry_id","candidate_entry_id"], how="inner", validate="one_to_one")
assert len(combined) == len(figure_scores)
print("Patent comparisons:", len(combined))
print("结论：每个瑞典专利对所有芬兰候选均得到原页区域分数。")
print("Conclusion: Every Swedish patent receives a page-region score against every Finnish candidate.")


# ## 5 · 融合裁剪图与原页证据 / Fuse crop and page evidence
# 
# 由于裁剪图分数和原页分数分布不同，先在每个瑞典查询内部做 z-score，再比较固定权重；同时加入基于排名的 RRF 融合。权重只用于探索，不能在这 89 对上选完后直接当成最终性能。
# 
# Because crop and page scores have different distributions, standardize each score within a Swedish query before fixed-weight fusion. Also test reciprocal-rank fusion. Weight selection here is exploratory and does not provide final independent performance.

# In[ ]:


def query_zscore(series):
    std = series.std(ddof=0)
    return (series - series.mean()) / (std if std > 1e-8 else 1.0)

combined["crop_z"] = combined.groupby("query_entry_id")["symmetric_best_mean"].transform(query_zscore)
combined["page_z"] = combined.groupby("query_entry_id")["page_region_symmetric_best"].transform(query_zscore)
combined["crop_rank"] = combined.groupby("query_entry_id")["symmetric_best_mean"].rank(method="first",ascending=False)
combined["page_rank"] = combined.groupby("query_entry_id")["page_region_symmetric_best"].rank(method="first",ascending=False)
combined["crop_only"] = combined["symmetric_best_mean"]
combined["page_only"] = combined["page_region_symmetric_best"]
combined["hybrid_page25"] = .75 * combined.crop_z + .25 * combined.page_z
combined["hybrid_page50"] = .50 * combined.crop_z + .50 * combined.page_z
combined["hybrid_page75"] = .25 * combined.crop_z + .75 * combined.page_z
combined["rrf_crop_page"] = 1/(60+combined.crop_rank) + 1/(60+combined.page_rank)
methods = ["crop_only","page_only","hybrid_page25","hybrid_page50","hybrid_page75","rrf_crop_page"]

metric_rows, rank_rows = [], []
for method in methods:
    ranked = combined[["query_entry_id","candidate_entry_id","is_known_pair",method]].copy()
    ranked["rank"] = ranked.groupby("query_entry_id")[method].rank(method="first",ascending=False).astype(int)
    truth = ranked[ranked.is_known_pair].copy()
    ranks = truth["rank"].to_numpy()
    metric_rows.append({"method":method,"pairs":len(ranks),"median_rank":float(np.median(ranks)),
                        "mrr":float(np.mean(1/ranks)),
                        **{f"recall_at_{k}":float(np.mean(ranks<=k)) for k in [1,3,5,10,20]}})
    truth["method"] = method
    rank_rows.append(truth.rename(columns={method:"true_pair_score"})[
        ["query_entry_id","candidate_entry_id","method","true_pair_score","rank"]])
metrics = pd.DataFrame(metric_rows).sort_values(["recall_at_1","recall_at_5","mrr"],ascending=False).reset_index(drop=True)
true_ranks = pd.concat(rank_rows,ignore_index=True)
display(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c not in ["method","pairs"]}))
print("Best exploratory fusion:", metrics.iloc[0].method)
print("结论：表中量化原页区域是否能补充分割图，并显示融合是否超过仅使用裁剪图。")
print("Conclusion: The table quantifies whether page regions recover segmentation evidence and whether fusion improves crop-only retrieval.")


# ## 6 · 检查分割问题案例 / Inspect segmentation-problem cases
# 
# 只查看人工反馈中标为局部组合、拆分或分割失败的专利，比较裁剪图、原页和最佳融合方法的正确专利排名。
# 
# For patents marked as partial/composite, fragmented, or segmentation failures in human feedback, compare the true-patent ranks from crop, page, and the best fusion method.

# In[ ]:


wide = true_ranks.pivot(index="query_entry_id",columns="method",values="rank")
problem_scopes = {"partial_composite_match","query_fragmented","query_split","segmentation_failure","candidate_partial"}
problem_entries = feedback.loc[feedback.match_scope.isin(problem_scopes),"query_entry_id"].drop_duplicates()
problem = wide.loc[wide.index.intersection(problem_entries)].copy()
best_method = metrics.iloc[0].method
problem["change_vs_crop"] = problem["crop_only"] - problem[best_method]
problem = problem.reset_index().sort_values("change_vs_crop",ascending=False)
display(problem[["query_entry_id","crop_only","page_only",best_method,"change_vs_crop"]])
print("结论：此表直接显示原页通道是否帮助了用户指出的分割失败样本。")
print("Conclusion: This table directly shows whether the page channel helps the user-identified segmentation failures.")


# ## 7 · 导出和结论 / Export and conclude
# 
# 导出区域索引、全部专利分数、评估指标、正确配对排名及分割问题案例。保留缓存后，后续 Run All 不会重复计算 DINOv3 embedding。
# 
# Export region indexes, all patent scores, metrics, known-pair ranks, and segmentation-problem cases. Once cached, later Run All executions do not recompute DINOv3 embeddings.

# In[ ]:


se_regions.to_csv(OUT/"sweden_page_region_manifest.csv",index=False,encoding="utf-8-sig")
fi_regions.to_csv(OUT/"finland_page_region_manifest.csv",index=False,encoding="utf-8-sig")
combined.to_csv(OUT/"crop_page_fusion_scores.csv",index=False,encoding="utf-8-sig")
metrics.to_csv(OUT/"crop_page_fusion_metrics.csv",index=False,encoding="utf-8-sig")
true_ranks.to_csv(OUT/"known_pair_crop_page_ranks.csv",index=False,encoding="utf-8-sig")
problem.to_csv(OUT/"segmentation_problem_rank_changes.csv",index=False,encoding="utf-8-sig")
summary = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model": MODEL_NAME, "regions_per_page": REGION_NAMES,
    "swedish_pages":len(se_pages),"finnish_pages":len(fi_pages),
    "swedish_regions":len(se_regions),"finnish_regions":len(fi_regions),
    "evaluable_known_pairs":int(combined.is_known_pair.sum()),"best_exploratory_method":best_method,
    "best_metrics":metrics.iloc[0].to_dict(),
    "input_sha256":{"sweden_pages":sha256(SWEDEN_PAGES),"finland_pages":sha256(FINLAND_PAGES),
                    "pooling_scores":sha256(POOLING_SCORES),"human_feedback":sha256(FEEDBACK_PATH)},
    "limitations":[
        "Fusion weights are explored on known pairs and require independent validation.",
        "Page embeddings may include labels and unrelated figures.",
        "Page regions supplement segmentation but do not replace crop-level evidence."
    ]
}
(OUT/"page_region_fallback_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
display(metrics)
print("结论：裁剪图与原页大区域的双通道检索已完成；结果可用于决定是否保留页面补充通道。")
print("Conclusion: Dual-channel crop and page-region retrieval is complete and can determine whether the page fallback should be retained.")

