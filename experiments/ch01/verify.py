"""Recompute the chapter's teaching examples; this is not a GPU benchmark."""
import json
import math
from pathlib import Path

DATA = json.loads(Path(__file__).with_name('cases.json').read_text(encoding='utf-8'))


def calculate():
    d = DATA
    weights = d['parameters'] * d['bytes_per_weight']
    weight_gib = weights / 2**30
    free = d['device_gib'] - d['working_gib'] - weight_gib
    per_request = (d['input_tokens'] + d['output_tokens']) * d['history_bytes_per_token'] / 2**30
    history = d['input_tokens'] * d['history_bytes_per_token']
    flop_count = 2 * d['parameters']
    bandwidth = d['memory_bytes_per_second']
    seq = d['sequential']
    service = seq['prefill_first_seconds'] + (d['output_tokens'] - 1) * seq['decode_seconds']
    batch = d['batch_size']
    return dict(
        weights_gib=weight_gib,
        history_budget_gib=free,
        request_reserved_gib=per_request,
        request_actual_final_gib=(d['input_tokens'] + d['output_tokens'] - 1) * d['history_bytes_per_token'] / 2**30,
        concurrency_memory=math.floor(free / per_request),
        total_for_16_gib=weight_gib+d['working_gib']+16*per_request,
        total_for_17_gib=weight_gib+d['working_gib']+17*per_request,
        training_states_gib=sum(d['training_state_bytes'])*d['parameters']/2**30,
        compute_ms=flop_count/d['compute_flops_per_second']*1000,
        weight_read_ms=weights/bandwidth*1000,
        full_read_ms=(weights+history)/bandwidth*1000,
        prefill_linear_seconds=flop_count*d['input_tokens']/d['compute_flops_per_second'],
        batch_read_gb=(weights+batch*history)/1e9,
        separate_read_gb=batch*(weights+history)/1e9,
        batch_read_ratio=batch*(weights+history)/(weights+batch*history),
        sequential_service_seconds=service,
        sequential_requests_per_second=1/service,
        second_wait_seconds=service-seq['arrival_interval_seconds'],
        second_first_latency=service+seq['prefill_first_seconds']-seq['arrival_interval_seconds'],
        client_end_seconds=d['request_trace_seconds']['first_visible']+(d['output_tokens']-1)*d['request_trace_seconds']['decode_interval'],
    )


def verify():
    d, r = DATA, calculate()
    assert math.isclose((2*1-6)**2/2, 8)
    w = 1 - .1 * (2*(2*1-6))
    assert math.isclose(w, 1.8) and math.isclose((2*w-6)**2/2, 2.88)
    assert r['concurrency_memory'] == 16
    assert r['total_for_16_gib'] < 24 < r['total_for_17_gib']
    for prompt, output, expected in [(8192,256,8), (8192,2048,7)]:
        assert math.floor(r['history_budget_gib']/((prompt+output)*d['history_bytes_per_token']/2**30)) == expected
    assert math.isclose(r['request_reserved_gib'], .53125)
    assert math.isclose(r['request_reserved_gib']-r['request_actual_final_gib'], 1/8192)
    assert math.isclose(r['compute_ms'], .1)
    assert math.isclose(r['weight_read_ms'], 12)
    assert math.isclose(r['full_read_ms'], 12.536870912)
    assert math.isclose(r['prefill_linear_seconds'], .4096)
    assert math.isclose(r['sequential_service_seconds'],6.975)
    assert math.isclose(r['second_wait_seconds'],4.975)
    assert math.isclose(r['second_first_latency'],5.575)
    assert math.isclose(r['client_end_seconds'],7.235)
    # Event-by-event FCFS simulation, independently of the closed form in the prose.
    end=0
    for j in range(10):
        arrival=j*d['sequential']['arrival_interval_seconds']
        begin=max(arrival,end)
        assert math.isclose(begin-arrival,j*(r['sequential_service_seconds']-2))
        end=begin+r['sequential_service_seconds']
    t=d['stalled_token_times']
    intervals=[b-a for a,b in zip(t,t[1:])]
    assert len(intervals)==9
    assert sum(math.isclose(v,.025) for v in intervals)==8
    assert math.isclose(max(intervals),.2)
    assert math.isclose(sum(intervals)/9,.4/9)
    train=[x['tokens_to_target']/x['tokens_per_second'] for x in d['training']]
    assert train==[120000,150000]
    assert train[1]-train[0]==30000
    assert 16000*train[0]==1920000000
    # A faster device changes a bound, not all unmodelled execution overhead.
    assert max(.05,12)==12 and max(.1,6)==6
    return r


if __name__ == '__main__':
    print(json.dumps(verify(),ensure_ascii=False,indent=2))
    print('All chapter 1 arithmetic and trace checks passed. No GPU timing was measured.')
