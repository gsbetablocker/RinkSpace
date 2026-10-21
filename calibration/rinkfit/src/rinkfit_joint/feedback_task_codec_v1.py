"""Lossless content-addressed feedback task documents; no source policy changes."""
from copy import deepcopy
import hashlib,json,math,os,re,struct,uuid
from pathlib import Path
SCHEMA='rinkfit.feedback_tasks.content_addressed.v1'
_HEX=re.compile(r'^[0-9a-f]{64}$')

def typed_sha256(value):
    """Bind JSON types and binary64, including signed zero, independently of dumps."""
    h=hashlib.sha256()
    def put(x):
        t=type(x)
        if x is None:h.update(b'n')
        elif t is bool:h.update(b't'if x else b'f')
        elif t is int:h.update(b'i'+str(x).encode('ascii')+b';')
        elif t is float:
            if not math.isfinite(x):raise ValueError('Nonfinite JSON number')
            h.update(b'd'+struct.pack('<d',x))
        elif t is str:
            z=x.encode('utf-8');h.update(b's'+str(len(z)).encode('ascii')+b':'+z)
        elif t is list:
            h.update(b'[')
            for y in x:put(y)
            h.update(b']')
        elif t is dict:
            if not all(type(k)is str for k in x):raise ValueError('Nonstring JSON object key')
            h.update(b'{')
            for k in sorted(x):put(k);put(x[k])
            h.update(b'}')
        else:raise ValueError('Unsupported JSON type: '+t.__name__)
    put(value);return h.hexdigest()

def assert_exact_json(left,right,path='$'):
    if type(left)is not type(right):raise ValueError('JSON type differs: '+path)
    if type(left)is float:
        if struct.pack('<d',left)!=struct.pack('<d',right):raise ValueError('binary64 differs: '+path)
    elif type(left)is dict:
        if set(left)!=set(right):raise ValueError('Object keys differ: '+path)
        for k in left:assert_exact_json(left[k],right[k],path+'.'+k)
    elif type(left)is list:
        if len(left)!=len(right):raise ValueError('List length differs: '+path)
        for i,(a,b)in enumerate(zip(left,right)):assert_exact_json(a,b,path+'['+str(i)+']')
    elif left!=right:raise ValueError('JSON value differs: '+path)

def encode(document):
    if type(document)is not dict or type(document.get('tasks'))is not list:raise ValueError('Feedback task document required')
    bodies={};matches={};tasks=[]
    for task in document['tasks']:
        if type(task)is not dict or type(task.get('cases'))is not list:raise ValueError('Task cases list required')
        refs=[]
        for wrapper in task['cases']:
            if type(wrapper)is not dict or 'case'not in wrapper:raise ValueError('Complete case wrapper required')
            body=wrapper['case'];bh=typed_sha256(body);mh=typed_sha256(wrapper)
            bodies.setdefault(bh,deepcopy(body))
            matches.setdefault(mh,dict(wrapper={k:deepcopy(v)for k,v in wrapper.items()if k!='case'},case_body_ref=bh))
            refs.append(mh)
        tasks.append(dict(task={k:deepcopy(v)for k,v in task.items()if k!='cases'},case_match_refs=refs))
    return dict(schema=SCHEMA,document={k:deepcopy(v)for k,v in document.items()if k!='tasks'},tasks=tasks,case_body_pool=bodies,case_match_pool=matches,decoded_document_typed_sha256=typed_sha256(document))

def decode(encoded):
    expected={'schema','document','tasks','case_body_pool','case_match_pool','decoded_document_typed_sha256'}
    if type(encoded)is not dict or set(encoded)!=expected or encoded['schema']!=SCHEMA:raise ValueError('Wrong content-addressed task schema/fields')
    if type(encoded['document'])is not dict or 'tasks'in encoded['document']:raise ValueError('Invalid document header')
    bodies=encoded['case_body_pool'];matches=encoded['case_match_pool']
    if type(bodies)is not dict or type(matches)is not dict or type(encoded['tasks'])is not list:raise ValueError('Invalid task pools')
    used_bodies=set();wrappers={}
    for key,body in bodies.items():
        if not _HEX.fullmatch(key)or typed_sha256(body)!=key:raise ValueError('Changed case body pool')
    for key,item in matches.items():
        if type(item)is not dict or set(item)!={'wrapper','case_body_ref'}or type(item['wrapper'])is not dict or 'case'in item['wrapper']:raise ValueError('Invalid case match pool entry')
        bh=item['case_body_ref']
        if type(bh)is not str or bh not in bodies:raise ValueError('Missing case body reference')
        wrapper=dict(item['wrapper'],case=bodies[bh])
        if not _HEX.fullmatch(key)or typed_sha256(wrapper)!=key:raise ValueError('Changed complete case match wrapper')
        used_bodies.add(bh);wrappers[key]=wrapper
    tasks=[];used_matches=set()
    for item in encoded['tasks']:
        if type(item)is not dict or set(item)!={'task','case_match_refs'}or type(item['task'])is not dict or 'cases'in item['task']or type(item['case_match_refs'])is not list:raise ValueError('Invalid encoded task')
        cases=[]
        for key in item['case_match_refs']:
            if type(key)is not str or key not in wrappers:raise ValueError('Missing case match reference')
            used_matches.add(key);cases.append(deepcopy(wrappers[key]))
        tasks.append(dict(deepcopy(item['task']),cases=cases))
    if used_matches!=set(matches)or used_bodies!=set(bodies):raise ValueError('Unreferenced pool content')
    result=dict(deepcopy(encoded['document']),tasks=tasks)
    if typed_sha256(result)!=encoded['decoded_document_typed_sha256']:raise ValueError('Decoded task document type/value identity differs')
    return result

def read_task_document(path):
    with Path(path).open(encoding='utf-8')as stream:value=json.load(stream)
    if type(value)is dict and value.get('schema')==SCHEMA:return decode(value)
    if type(value)is not dict or type(value.get('tasks'))is not list:raise ValueError('Unsupported feedback task document')
    return value

def write_task_document(path,document):
    """Compact atomic output; never materialize a hydrated task file."""
    from rinkspace.path_policy import guard_project_write
    path=guard_project_write(path)
    if path.exists():raise FileExistsError('Immutable task generation exists: '+str(path))
    encoded=encode(document);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=guard_project_write(path.with_name(path.name+'.'+uuid.uuid4().hex[:12]+'.tmp'))
    with temporary.open('x',encoding='utf-8',newline='\n')as stream:
        json.dump(encoded,stream,ensure_ascii=False,separators=(',',':'),allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    if path.exists():raise FileExistsError('Task generation appeared during write')
    os.replace(temporary,path)
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),schema=SCHEMA,decoded_document_typed_sha256=encoded['decoded_document_typed_sha256'],task_count=len(encoded['tasks']),case_bodies=len(encoded['case_body_pool']),case_matches=len(encoded['case_match_pool']))
