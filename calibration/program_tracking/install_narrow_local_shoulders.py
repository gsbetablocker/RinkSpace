from pathlib import Path
p=Path('calibration/program_tracking/native_ice_material_repair.py');s=p.read_text();a=s.index("    if group['outline']:\n",s.index(' def measurements'));b=s.index('    else:row=native_single_colour_edge',a)
s=s[:a]+'''    if group['outline']:
     choices=narrow_dark_paint(q[j],n[j],offsets[j],profiles[j],validity[j])
     choices.sort(key=lambda r:abs((np.asarray(r['uv'])-q[j])@n[j]))
     row=choices[0] if choices else None
''' +s[b:]
s=s.replace("if len(rows)<6:return {},diagnostics","if len(rows)<(3 if fid=='C0' else 6):return {},diagnostics")
s=s.replace("if len(selected)<(5 if group['outline'] else 3):continue","if len(selected)<3:continue")
p.write_text(s)
