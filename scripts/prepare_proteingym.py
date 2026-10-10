"""Freeze a ProteinGym target-sequence panel and three complete DMS assays.

Selection uses metadata only, never model scores or experimental fitness values.
"""
import argparse
import csv
import datetime
import json
from pathlib import Path
import shutil
import sys
import zipfile
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from common.reporting import write_json,sha256
COMMIT='144fe22b07dfaeec2b366f2346203a9838a55b4c'
ASSAYS=['RL40A_YEAST_Mavor_2016','BLAT_ECOLX_Firnberg_2014','SPG1_STRSG_Wu_2016']
BINS=[(1,128),(129,256),(257,512),(513,1024),(1025,2048)]
AMINO=set('ACDEFGHIKLMNPQRSTVWY')


def panel(rows):
    unique={}
    excluded=[]
    for row in sorted(rows,key=lambda r:r['DMS_id']):
        seq=row['target_seq']
        if len(seq)!=int(row['seq_len']):raise ValueError('Metadata sequence length mismatch')
        if not seq or not set(seq)<=AMINO:
            excluded.append({'assay':row['DMS_id'],'reason':'noncanonical sequence'});continue
        if len(seq)>2048:
            excluded.append({'assay':row['DMS_id'],'reason':'outside fixed current <=2048 length contract'});continue
        unique.setdefault(row['UniProt_ID'],row)
    taxa=['Human','Eukaryote','Prokaryote','Virus'];selected=[];selected_hashes=set()
    for i,(low,high) in enumerate(BINS):
        candidates=sorted([r for r in unique.values() if low<=len(r['target_seq'])<=high],key=lambda r:r['UniProt_ID'])
        preferred=taxa[2:] + taxa[:2] if i%2 else taxa
        choices=[]
        for taxon in preferred:
            candidate=next((r for r in candidates if r['taxon']==taxon and r['UniProt_ID'] not in {x['UniProt_ID'] for x in choices}
                            and r['target_seq'] not in selected_hashes),None)
            if candidate:
                choices.append(candidate);selected_hashes.add(candidate['target_seq'])
            if len(choices)==2:break
        if len(choices)!=2:raise ValueError(f'Insufficient distinct targets in length bin {low}-{high}')
        selected.extend({'id':'pg_'+r['UniProt_ID'],'sequence':r['target_seq'],'uniprot':r['UniProt_ID'],
            'taxon':r['taxon'],'length_bin':[low,high],'source_assay':r['DMS_id'],
            'source':'ProteinGym v1.3 assay target sequence (may be domain or engineered target, not necessarily full UniProt protein)',
            'description':r.get('protein_name',r.get('molecule_name',''))} for r in choices)
    return selected,excluded


def main(a):
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    rows=list(csv.DictReader(Path(a.reference).open()))
    selected,excluded=panel(rows)
    shutil.copy2(a.reference,root/'DMS_substitutions.csv')
    (root/'performance_panel.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    manifest={'dataset':'ProteinGym v1.3','reference_commit':COMMIT,'downloaded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'sources':{'reference':f'https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/{COMMIT}/reference_files/DMS_substitutions.csv',
                  'data':'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip',
                  'upstream':'https://github.com/OATML-Markslab/ProteinGym'},
       'reference_sha256':sha256(root/'DMS_substitutions.csv'),'archive_sha256':sha256(a.archive),
       'archive_bytes':Path(a.archive).stat().st_size,'hash_note':'Locally calculated; zip member CRC is also checked, not a publisher-signed digest',
       'reference_assays':len(rows),'reference_unique_proteins':len({r['UniProt_ID'] for r in rows}),
       'performance_panel':selected,'excluded':excluded,
       'selection_policy':'Two distinct target sequences/proteins in each frozen length bin <=2048; sorted IDs and alternating taxon preference; no outcome-based selection',
       'quality_assays':[], 'limitations':['Bounded engineering/quality panel, not full 217-assay leaderboard',
        'No verified training-data isolation; zero-shot means no label fitting here, not guaranteed absence from pretraining',
        'Assay targets can be domains or engineered proteins; taxonomy and length alone do not prove population representativeness',
        'OpenFold structure benchmark is a separate requirement'],
       'preparer_sha256':sha256(__file__)}
    with zipfile.ZipFile(a.archive) as archive:
        for identifier in ASSAYS:
            row=next(r for r in rows if r['DMS_id']==identifier)
            matches=[n for n in archive.namelist() if Path(n).name==row['DMS_filename']]
            if len(matches)!=1:raise ValueError('Expected exactly one official assay file: '+identifier)
            path=root/row['DMS_filename']
            path.write_bytes(archive.read(matches[0]))  # CRC verified; never extract arbitrary paths.
            assay_rows=list(csv.DictReader(path.open()))
            if len(assay_rows)!=int(row['DMS_total_number_mutants']):raise ValueError('Assay row count differs from pinned metadata')
            manifest['quality_assays'].append({'id':identifier,'metadata':row,'path':str(path),
                'member':matches[0],'rows':len(assay_rows),'sha256':sha256(path)})
    (root/'SOURCE_LICENSE.md').write_text('ProteinGym code/dataset repository: MIT. Preserve original experimental citations in each metadata record.\nSource: https://github.com/OATML-Markslab/ProteinGym\n')
    write_json(root/'manifest.json',manifest)
    print('FROZEN',root,[(r['uniprot'],len(r['sequence']),r['taxon']) for r in selected])
    print('Assays',[(r['id'],r['rows']) for r in manifest['quality_assays']])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True)
    p.add_argument('--archive',required=True);p.add_argument('--output',required=True);main(p.parse_args())
