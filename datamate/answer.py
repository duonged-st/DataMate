"""DataMate JSONL adapter, ported from the V4.2 notebook."""

import json
import math
import re
import unicodedata
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from numbers import Number

ALLOWED_TYPES = {'number', 'percent', 'date', 'boolean', 'string', 'list', 'table',
                 'unanswerable', 'abstain'}

def numeric_json(v, kind, q):
    if isinstance(v, bool) or not isinstance(v, Number):
        raise ValueError('Đáp án kiểu ' + kind + ' phải là JSON number')
    x = float(v)
    if not math.isfinite(x):
        raise ValueError('NaN/Inf không hợp lệ với JSON')
    qtext = str(q['question']).casefold()
    # Các chỉ tiêu tiền phải làm tròn về VND, half-up; SQL nên tự làm tròn đúng.
    money_terms = ('doanh thu', 'tiền', 'vnd', 'đồng', 'giá trị', 'giá bán',
                   'giá nhập', 'lãi', 'nợ', 'thu về', 'bao nhiêu vốn', 'tổng bán')
    if kind == 'percent':
        return float(Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    if any(t in qtext for t in money_terms):
        return int(Decimal(str(x)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    if x.is_integer():
        return int(x)
    return float(Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def clean_nested_json(obj):
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, Number):
        x=float(obj)
        if not math.isfinite(x):
            raise ValueError('NaN/Inf trong kết quả')
        if x.is_integer():
            return int(x)
        return x
    if isinstance(obj, str):
        return unicodedata.normalize('NFC', obj.strip())
    if isinstance(obj, list):
        return [clean_nested_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [clean_nested_json(x) for x in obj]
    raise ValueError('Không chuyển được sang giá trị JSON hợp lệ: ' + str(type(obj)))


def btc_answer_row(q, answer_type, raw_value, reason=None):
    # NEVER access answers / gold type here.
    row = {'id':q['id'], 'type':answer_type}
    if answer_type == 'abstain':
        return row
    if answer_type == 'unanswerable':
        # Reason được BTC cho phép nhưng không bắt buộc; bỏ trống cho an toàn.
        return row
    if answer_type not in ALLOWED_TYPES:
        raise ValueError('Invalid type: ' + str(answer_type))

    if answer_type in ('number','percent'):
        ans = numeric_json(raw_value, answer_type, q)
    elif answer_type == 'boolean':
        if isinstance(raw_value, bool):
            ans = raw_value
        elif isinstance(raw_value, (int,float)) and raw_value in (0,1):
            ans = bool(raw_value)
        else:
            raise ValueError('Boolean phải là 0/1 hoặc bool')
    elif answer_type == 'date':
        if not isinstance(raw_value,str):
            raise ValueError('Date phải là chuỗi YYYY-MM-DD / YYYY-MM')
        ans = raw_value.strip()
        if not re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?',ans):
            raise ValueError('Ngày không ở format chuẩn: ' + ans)
        date.fromisoformat(ans + '-01' if len(ans) == 7 else ans)
    elif answer_type == 'string':
        if not isinstance(raw_value,str):
            raise ValueError('String phải là một tên/chuỗi')
        ans = unicodedata.normalize('NFC',raw_value.strip())
    elif answer_type == 'list':
        if isinstance(raw_value,list):
            ans = clean_nested_json(raw_value)
        elif isinstance(raw_value,dict):
            raw_rows = raw_value.get('rows',[])
            if len(raw_rows)==1:
                ans = clean_nested_json(raw_rows[0])
            elif all(len(r)==1 for r in raw_rows):
                ans = clean_nested_json([r[0] for r in raw_rows])
            else:
                raise ValueError('Nhiều dòng/cột không suy ra list có thứ tự')
        elif isinstance(raw_value,(str,Number)):
            ans = [clean_nested_json(raw_value)]
        else:
            raise ValueError('Không chuyển được sang list')
    else: # table
        if not isinstance(raw_value,dict):
            raise ValueError('Table cần SQL trả nhiều cột')
        ans = {'columns': [str(c) for c in raw_value['columns']],
               'rows': [clean_nested_json(row_values) for row_values in raw_value['rows']]}
        if len(ans['columns']) < 2 or any(len(r)!=len(ans['columns']) for r in ans['rows']):
            raise ValueError('Table cần ít nhất 2 cột; mỗi row phải đồng dạng')

    # json.dumps(..., allow_nan=False) catches remaining invalid numbers
    json.dumps(ans, ensure_ascii=False, allow_nan=False)
    row['answer']=ans
    return row


def validate_btc_submission(rows, expected_ids=None):
    ids=[]
    for row in rows:
        if not isinstance(row,dict) or row.get('type') not in ALLOWED_TYPES:
            raise ValueError('Dòng submission không hợp lệ: ' + str(row))
        if set(row) - {'id','type','answer','reason'}:
            raise ValueError('Submission bị lẫn diagnostic/gold: ' + str(row))
        if not isinstance(row.get('id'),str) or not row['id']:
            raise ValueError('ID không hợp lệ')
        typ=row['type']
        if typ in {'abstain','unanswerable'} and 'answer' in row:
            raise ValueError('Không được có answer cho abstain/unanswerable')
        if typ not in {'abstain','unanswerable'} and 'answer' not in row:
            raise ValueError('Thiếu answer')
        if 'reason' in row and not isinstance(row['reason'],str):
            raise ValueError('reason nếu có phải là string')
        json.dumps(row,ensure_ascii=False,allow_nan=False)
        ids.append(row['id'])
    if len(ids)!=len(set(ids)):
        raise ValueError('Submission có ID trùng')
    if expected_ids is not None and set(ids)!=set(expected_ids):
        raise ValueError('Submission thiếu/thừa ID so với bộ benchmark')
    return True


