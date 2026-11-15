"""Rebuild the CPU smoke evidence from the supplied BTC Public Test split."""

import argparse
import json
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from datamate.answer import validate_btc_submission
from datamate.data import Dataset, build_business, quote
from datamate.inference import execute_sql
from datamate.normalize import as_number
from datamate.prompt import SplitContext

ROOT = Path(__file__).resolve().parent


def smoke(data, cache, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    checks=[]
    with Dataset(data) as dataset:
        qs=dataset.jsonl('questions.jsonl')
        context=SplitContext(dataset)
        assert context.dates({'question':'Doanh thu dịp Tết 2026?','as_of_date':'2026-03-01'}).find('2026-02-10') >= 0
        assert as_number('1tr440','money')==1440000
        samples={}
        for q in qs:
            samples.setdefault(q['business_id'], q)
        expectations={}
        for biz,q in samples.items():
            db,cat,readme=build_business(dataset,biz,cache)
            table=cat['tables'][0]
            expectation=table['rows']
            assert execute_sql(db,f'SELECT COUNT(*) FROM {quote(table["name"])}')["rows"]==[[expectation]]
            expectations[q['question']]=(table['name'],expectation,q['id'])
            assert readme
        db,cat,_=build_business(dataset,'dev_001',cache)
        result=execute_sql(db, 'SELECT COUNT(*) FROM so_chi_tiet_ban_hang WHERE typeof("Thành tiền") != \'integer\'')
        assert result["rows"] == [[0]],result
        try:
            execute_sql(db, 'SELECT load_extension(\'disabled\')')
        except sqlite3.DatabaseError:
            checks.append('SQLite readonly authorization')
        else:
            raise AssertionError('Readonly policy did not reject extension')
        db,cat,_=build_business(dataset,'dev_005',cache)
        units=execute_sql(db,'''SELECT factor FROM unit_factors WHERE product_name LIKE '%Panadol Extra%' AND unit='hộp' ''')
        assert [180] in units["rows"],units

    class Fixtures(BaseHTTPRequestHandler):
        calls={}
        def send_json(self, payload):
            content=json.dumps(payload,ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        def do_GET(self):
            assert self.path=='/v1/models'
            self.send_json({'data':[{'id':'cpu-fixture'}]})
        def do_POST(self):
            assert self.path=='/v1/chat/completions'
            obj=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            message=obj['messages'][-1]['content']
            assert 'README của nguồn dữ liệu' in message
            assert 'Quy tắc của bộ dữ liệu' in message
            question=message.splitlines()[0].removeprefix('Câu hỏi: ')
            name,n,identifier=expectations[question]
            Fixtures.calls[identifier]=Fixtures.calls.get(identifier,0)+1
            # Exercise SQL repair in one question. Use a qualified missing column because SQLite
            # accepts double-quoted unknown unqualified identifiers as string literals.
            if identifier==first_id and Fixtures.calls[identifier]==1:
                plan={'type':'number','sql': f'SELECT t.nonexistent FROM {quote(name)} t'}
            else:
                if identifier==first_id:
                    assert 'Sửa câu trả lời trước' in message
                plan={'type':'number','sql': f'SELECT COUNT(*) FROM {quote(name)}'}
            self.send_json({'choices':[{'message':{'content':json.dumps(plan)}}]})
        def log_message(self,*_): pass

    first_id=next(iter(samples.values()))['id']
    with TemporaryDirectory() as work, ThreadingHTTPServer(('127.0.0.1',0),Fixtures) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        work=Path(work)
        manifest=work/'smoke_ids.jsonl'
        manifest.write_text(''.join(json.dumps({'id':q['id']})+'\n' for q in samples.values()))
        try:
            subprocess.run([sys.executable,'-m','datamate','predict','--data',str(data),'--questions-file',str(manifest),
                            '--work',str(cache),'--output',str(work),'--endpoint',f'http://127.0.0.1:{server.server_port}'],check=True, cwd=ROOT)
        finally:
            server.shutdown()
        rows=[json.loads(x) for x in (work/'submission.jsonl').read_text().splitlines()]
        logs=[json.loads(x) for x in (work/'logs'/'questions.jsonl').read_text().splitlines()]
        validate_btc_submission(rows,[q['id'] for q in samples.values()])
        subprocess.run([sys.executable,str(ROOT / "verify_outputs.py"),"--data",str(data),
                        "--output",str(work),"--compare",str(work),"--questions-file",str(manifest)],check=True, cwd=ROOT)
        expected={identifier:n for _,n,identifier in expectations.values()}
        for row, log in zip(rows, logs, strict=True):
            assert row['id'] == log['id'] and log['elapsed_ms'] >= 0 and log['thinking'] is None
            if row['id'] in Fixtures.calls:
                assert row['type'] == 'number' and row['answer'] == expected[row['id']]
                assert log['code'] and log['output']
            elif row['type'] == 'unanswerable':
                assert any(a.get('source') == 'catalog_evidence' for a in log['attempts'])
            else:
                assert log['code'] and log['output']
                assert any(a.get('source') == 'rulebook_sql' and 'error' not in a for a in log['attempts'])
        assert len(logs) == len(samples) and Fixtures.calls[first_id] == 2

    report={'passed':True,'businesses':len(samples),'answers_validated':len(rows),'logs_validated':len(logs),'model_calls_simulated':sum(Fixtures.calls.values()),
            'checks':checks+['valid BTC output', 'repair after SQL error', 'Excel mixed-locale currency', 'chained unit factors', 'Tết period', 'README and split rules in prompt'],
            'backend':'Scripted localhost endpoint returning SQL fixtures. Scores validate execution and formatting only. Real Qwen and GPU Docker are pending.'}
    output.write_text(json.dumps(report,indent=2))
    print(report)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--work", type=Path, default=Path(".datamate_cache"))
    parser.add_argument("--output", type=Path, default=ROOT / "verification/cpu_smoke.json")
    args = parser.parse_args()
    smoke(args.data.resolve(), args.work.resolve(), args.output.resolve())
