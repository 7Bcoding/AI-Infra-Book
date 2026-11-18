"""Run: python evaluation_protocol.py [record.json]. Standard-library only.

Default records are explicitly synthetic. This validates accounting and computes
metrics; it does not establish measurement independence or causal improvement.
"""
import argparse
from copy import deepcopy
from itertools import combinations
import json
from math import ceil, comb, isclose, sqrt
from pathlib import Path
from statistics import mean, stdev


def by_id(rows):
    result={r['id']:r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate request ID')
    return result


def cohort(case):
    planned=by_id(case['scheduled']); results=by_id(case['outcomes'])
    if planned.keys()!=results.keys():raise ValueError('Every scheduled ID needs an outcome, including pending/failed')
    w=case['window']; start=w['start_ms']; stop=w['stop_injection_ms']; observe=w['observe_until_ms']
    if not start<stop<=observe:raise ValueError('Invalid measurement window')
    latencies=[];good=unknown=0
    for request_id,row in planned.items():
        arrival=row['arrival_ms']; r=results[request_id]
        if not start<=arrival<stop:raise ValueError('Arrival outside half-open injection window')
        status=r['status']; finish=r['finish_ms'];sent=r['sent_ms']
        if status not in {'success','failure','timeout','rejected','pending'}:raise ValueError('Unknown status')
        if sent is not None and sent<arrival:raise ValueError('Sending precedes scheduled arrival')
        if finish is not None and (finish<arrival or (sent is not None and finish<sent)):raise ValueError('Invalid completion time')
        if status!='pending' and finish is None:raise ValueError('Terminal outcome has no timestamp')
        if status=='success' and sent is None:raise ValueError('Success has no send timestamp')
        if status=='pending' or finish is None or finish>observe:
            unknown+=1;continue
        if status=='success':
            latency=finish-arrival;latencies.append(latency)
            good+=int(r['correct'] and latency<=w['slo_ms'])
    n=len(planned)
    if not n:raise ValueError('Empty cohort')
    ordered=sorted(latencies)
    return dict(scheduled=n,observed_success=len(ordered),pending=unknown,good=good,
                good_fraction_lower=good/n,good_fraction_upper=(good+unknown)/n,
                observed_success_p99_ms=ordered[ceil(.99*len(ordered))-1] if ordered else None,
                good_per_injection_second=good/((stop-start)/1000),
                good_per_observation_second=good/((observe-start)/1000))


def paired(case):
    values=[]
    for r in case['paired_windows']:
        if min(r['old_ms'],r['new_ms'])<=0:raise ValueError('Nonpositive duration')
        values.append(1-r['new_ms']/r['old_ms'])
    if len(values)<2:raise ValueError('Need repeated paired windows')
    result=dict(n=len(values),mean_saved_fraction=mean(values),sample_sd=stdev(values),standard_error=stdev(values)/sqrt(len(values)))
    # The published worked example has df=4. Do not reuse its critical value
    # for another sample size or silently claim distribution-free inference.
    if len(values)==5:
        half=2.776*result['standard_error']
        result['normal_independent_windows_t95']=[result['mean_saved_fraction']-half,result['mean_saved_fraction']+half]
    return result


def dominates(a,b):
    av=(-a['good_tasks_s'],a['p99_ms'],a['joules_per_good_task'])
    bv=(-b['good_tasks_s'],b['p99_ms'],b['joules_per_good_task'])
    return all(x<=y for x,y in zip(av,bv)) and any(x<y for x,y in zip(av,bv))


def energy(case):
    rows=case['power_samples'];total=0
    if len(rows)<2:raise ValueError('Power integration needs time intervals')
    for a,b in zip(rows,rows[1:]):
        if b['t_s']<=a['t_s'] or min(a['watts'],b['watts'])<0:raise ValueError('Invalid power samples')
        total+=(b['t_s']-a['t_s'])*(a['watts']+b['watts'])/2
    count=case['good_tasks_in_energy_window']
    if count<=0:raise ValueError('No good tasks to normalize energy')
    return dict(joules=total,joules_per_good_task=total/count,scope=case['power_scope'],integration='piecewise_linear')


def pass_at_k(n,c,k):
    if not (0<=c<=n and 1<=k<=n):raise ValueError('Invalid candidate counts')
    return 1-(comb(n-c,k)/comb(n,k) if n-c>=k else 0)


def report(case):
    eligible=[x for x in case['candidates'] if x['p99_ms']<=150]
    frontier=[x['name'] for x in eligible if not any(dominates(y,x) for y in eligible)]
    return dict(data_kind=case['kind'],cohort=cohort(case),paired=paired(case),
                feasible_nondominated=frontier,energy=energy(case))


def verify(case):
    r=report(case);c=r['cohort']
    assert c['scheduled']==6 and c['good']==5 and c['pending']==0
    assert c['observed_success_p99_ms']==22
    assert isclose(c['good_per_injection_second'],5/.06)
    assert isclose(c['good_per_observation_second'],5/.072)
    truncated=deepcopy(case);truncated['window']['observe_until_ms']=60
    early=cohort(truncated)
    assert early['pending']==1 and early['observed_success_p99_ms']==20
    failed=deepcopy(case);failed['outcomes'][0]['status']='failure'
    assert cohort(failed)['good']==4 and cohort(failed)['scheduled']==6
    missing=deepcopy(case);missing['outcomes'].pop()
    try:cohort(missing)
    except ValueError:pass
    else:raise AssertionError('Missing request was silently dropped')
    duplicate=deepcopy(case);duplicate['outcomes'].append(duplicate['outcomes'][0])
    try:cohort(duplicate)
    except ValueError:pass
    else:raise AssertionError('Duplicate request was silently accepted')
    assert r['feasible_nondominated']==['A','B']
    assert r['energy']['joules_per_good_task']==12
    assert isclose(r['paired']['mean_saved_fraction'],.10)
    low,high=r['paired']['normal_independent_windows_t95']
    assert .060<low<.061 and .139<high<.140
    exhaustive=mean(any(i<3 for i in pair) for pair in combinations(range(10),2))
    assert isclose(pass_at_k(10,3,2),exhaustive)
    assert isclose(exhaustive,24/45)
    assert pass_at_k(10,10,2)==1 and pass_at_k(10,0,2)==0
    assert 200+100+50+300==650 and 100+100+50+150==400
    assert 400+100+75==575
    print('Verified missing/failed requests, accounting boundaries, paired statistics, energy and pass@k.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record',nargs='?',type=Path)
    args=parser.parse_args()
    path=args.record or Path(__file__).with_name('evaluation_case.json')
    case=json.loads(path.read_text(encoding='utf-8'))
    if args.record is None:verify(case)
    print(json.dumps(report(case),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
