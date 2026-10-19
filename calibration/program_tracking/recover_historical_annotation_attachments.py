"""Recover missing user annotation images from their own local chat events."""
from __future__ import annotations
import base64, hashlib, io, json, re
from pathlib import Path
from PIL import Image
from .finalize_first_shot_review import OUT, DATA, read, write

def main():
    path=OUT/'historical_manual_annotation_inventory.json';inventory=read(path)
    missing={Path(r['original_path']).name for r in inventory['historical_user_image_feedback'] if not r.get('available_path') and not r.get('linked_registered_source')}
    if not missing:
        print(json.dumps(dict(status='already_recovered',image_count=inventory['counts'].get('recovered_chat_image_copies_from_missing_temp_paths',0))))
        return
    threads={'private-thread-00008','private-thread-00009'}
    folder=DATA/'user_annotations/recovered_session_images';folder.mkdir(parents=True,exist_ok=True)
    recovered={}
    for session in Path('D:/CodexData/.codex/sessions/2026/10/02').glob('*.jsonl'):
        thread=next((t for t in threads if t in session.name),None)
        if thread is None:continue
        with session.open(encoding='utf8') as handle:
            for number,line in enumerate(handle,1):
                event=json.loads(line);payload=event.get('payload',{});content=payload.get('content',[])
                if payload.get('role')!='user' or not isinstance(content,list):continue
                text='\n'.join(item.get('text','') for item in content if isinstance(item,dict))
                names=list(dict.fromkeys(re.findall(r'codex-clipboard-[0-9a-f-]+\.(?:png|jpg|jpeg)',text)))
                images=[item for item in content if isinstance(item,dict) and item.get('type')=='input_image']
                if not set(names)&missing:continue
                if len(names)!=len(images):raise ValueError('Ambiguous annotation/image ordering')
                for name,image in zip(names,images):
                    if name not in missing:continue
                    prefix,encoded=image['image_url'].split(',',1)
                    if not prefix.startswith('data:image/') or not prefix.endswith(';base64'):raise ValueError('Unexpected image reference')
                    raw=base64.b64decode(encoded,validate=True);image_path=folder/name
                    if image_path.exists() and image_path.read_bytes()!=raw:raise ValueError('Conflicting recovered image')
                    image_path.write_bytes(raw)
                    with Image.open(io.BytesIO(raw)) as actual:width,height=actual.size;fmt=actual.format
                    recovered[name]=dict(path=str(image_path),sha256=hashlib.sha256(raw).hexdigest(),byte_count=len(raw),width=width,height=height,image_format=fmt,
                        source_session=str(session),source_event_line_1based=number,source_event_timestamp=event.get('timestamp'),thread_id=thread,
                        source_event_sha256=hashlib.sha256(line.encode('utf8')).hexdigest(),copy_kind='verbatim_chat_input_image_bytes_not_unavailable_temp_original',
                        original_temp_bytes_claimed=False,pixel_registration_required_before_native_frame_use=True)
    if missing-set(recovered):raise ValueError('Unrecovered annotations: '+str(missing-set(recovered)))
    for row in inventory['historical_user_image_feedback']:
        name=Path(row['original_path']).name
        if name in recovered:
            item=recovered[name];row.update(available_path=item['path'],sha256=item['sha256'],available_path_sha_verified=True,
                recovered_from_local_chat_input_image=True,available_copy_kind=item['copy_kind'],chat_image_native_size=[item['width'],item['height']],
                session_recovery_receipt=str(DATA/'user_annotations/session_image_recovery.json'))
    inventory['counts'].update(recovered_chat_image_copies_from_missing_temp_paths=len(recovered),historical_chat_images_unrecoverable=0,
        recovered_additional_historical_images=inventory['counts']['recovered_additional_historical_images']+len(recovered))
    inventory['search_limitations']=[line for line in inventory['search_limitations'] if not line.startswith('10张更早的Temp')]
    inventory['search_limitations'].append('10张更早的Temp原文件失效，但已全部从对应本地聊天的input_image恢复原样图像副本；这些是聊天所存图像，不冒充失效Temp原件或原生视频像素。')
    inventory['session_image_recovery']=dict(path=str(DATA/'user_annotations/session_image_recovery.json'),image_count=len(recovered))
    write(DATA/'user_annotations/session_image_recovery.json',dict(schema='rinkspace.user-chat-annotation-image-recovery.v1',images=list(recovered.values()),missing_count=0))
    write(path,inventory)
    print(json.dumps(dict(recovered=len(recovered),missing_count=0,images=[dict(name=k,width=v['width'],height=v['height']) for k,v in recovered.items()]),ensure_ascii=False))

if __name__=='__main__':main()
