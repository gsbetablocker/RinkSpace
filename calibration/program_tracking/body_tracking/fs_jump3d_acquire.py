"""Read public dataset ZIP members with HTTP byte ranges, preserving source IDs.

No account credentials, model execution or dataset-wide download is performed.
Extracted bytes are checked against the original ZIP CRC by zipfile.
"""
from __future__ import annotations
import argparse,hashlib,io,json,re,time,urllib.request,zipfile
from pathlib import Path


class RangeFile(io.RawIOBase):
    def __init__(self,url,*,timeout=45):
        self.url=url;self.timeout=timeout;self.position=0;self.requests=0;self.transferred=0
        self.size=None;self._cache={};self._buffer=(0,b'')
        self._fetch(0,0)

    def _fetch(self,start,end):
        key=(start,end)
        if key in self._cache:return self._cache[key]
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','User-Agent':'RinkSpace-dataset-validation/1'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request,timeout=self.timeout)as response:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if response.status!=206 or not match:raise IOError('Server must supply explicit HTTP 206 Content-Range')
                    first,last,total=map(int,match.groups())
                    if first!=start or last!=end:raise IOError('Returned range differs from requested byte range')
                    if self.size is not None and self.size!=total:raise IOError('Remote archive size changed')
                    self.size=total;data=response.read()
                    if len(data)!=end-start+1:raise IOError('Truncated remote byte range')
                    self.requests+=1;self.transferred+=len(data)
                    if len(data)<=1024*1024:self._cache[key]=data
                    return data
            except (OSError,TimeoutError):
                if attempt==2:raise
                time.sleep(1+attempt)

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if position<0:raise ValueError('Negative seek')
        self.position=position;return position
    def read(self,size=-1):
        if size<0:size=max(0,self.size-self.position)
        size=min(size,max(0,self.size-self.position))
        if not size:return b''
        start,data=self._buffer;offset=self.position-start
        if 0<=offset and offset+size<=len(data):
            result=data[offset:offset+size]
        else:
            # ZIP local header/name/extra/payload reads are adjacent. One
            # bounded prefetch avoids a round trip for every tiny field.
            stop=min(self.size,self.position+max(size,256*1024))
            data=self._fetch(self.position,stop-1);self._buffer=(self.position,data);result=data[:size]
        self.position+=size;return result


def download_url(file_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',file_id):raise ValueError('Invalid public Google Drive ID')
    return 'https://drive.usercontent.google.com/download?id='+file_id+'&export=download&confirm=t'


def archive_index(url):
    remote=RangeFile(url)
    with zipfile.ZipFile(remote)as archive:
        members=[dict(name=i.filename,size=i.file_size,compressed_size=i.compress_size,CRC32=f'{i.CRC:08x}',compression=i.compress_type)for i in archive.infolist()]
    return dict(url=url,size=remote.size,bytes_transferred=remote.transferred,http_requests=remote.requests,members=members)


def extract_members(url,names,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);remote=RangeFile(url);records=[]
    with zipfile.ZipFile(remote)as archive:
        for name in names:
            relative=Path(name)
            target=(output/relative).resolve()
            if not target.is_relative_to(output)or relative.is_absolute():raise ValueError('Unsafe archive path')
            info=archive.getinfo(name)
            if info.is_dir():continue
            target.parent.mkdir(parents=True,exist_ok=True);hasher=hashlib.sha256()
            if target.exists():
                data=target.read_bytes()
                import zlib
                if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:raise FileExistsError('Existing bytes differ '+str(target))
                hasher.update(data);reused=True
            else:
                pending=target.with_name(target.name+'.partial')
                with archive.open(info)as source,pending.open('wb')as sink:
                    while block:=source.read(1024*1024):sink.write(block);hasher.update(block)
                if pending.stat().st_size!=info.file_size:raise IOError('Extracted member size mismatch')
                pending.replace(target);reused=False
            record=dict(member=name,path=str(target),size=info.file_size,sha256=hasher.hexdigest(),zip_CRC32=f'{info.CRC:08x}',CRC_checked=True,reused=reused)
            records.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
    return dict(url=url,archive_size=remote.size,bytes_transferred=remote.transferred,http_requests=remote.requests,members=records)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--file-id',required=True);ap.add_argument('--index',type=Path);ap.add_argument('--members',nargs='*');ap.add_argument('--output',type=Path);a=ap.parse_args();url=download_url(a.file_id)
    if a.index:
        value=archive_index(url)
        with a.index.open('x',encoding='utf-8')as out:json.dump(value,out,ensure_ascii=False,indent=2)
        print(json.dumps(dict(index=str(a.index),archive_bytes=value['size'],members=len(value['members']),transferred=value['bytes_transferred'])))
    if a.members:
        if not a.output:ap.error('--output required for extraction')
        extract_members(url,a.members,a.output)


if __name__=='__main__':main()
