#!/usr/bin/env python3
"""HotpotQA sentence-level selection. Python 3.10+, standard library only."""
import argparse
import csv
import hashlib
import json
import platform
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import mean, median
from time import perf_counter_ns

TOKEN = re.compile(r'[A-Za-z0-9]+')
STOP = frozenset('a an the of to in on at by for with from and or is are was were be been being do does did what which who whom whose where when why how this that these those it its he she his her their they them as than have has had'.split())
BUDGETS = (60, 120, 240)
OFFICIAL_SHA256 = '4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea'


def tokens(text):
    return TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Unit:
    title: str
    sentence_id: int
    sentence: str
    length: int
    mask: int


def prepare(question, context):
    q = list(dict.fromkeys(w for w in tokens(question) if w not in STOP and (len(w) >= 3 or w.isdigit())))[:12]
    units = []
    for title, sentences in context:
        for sid, sentence in enumerate(sentences):
            text = title + ' ' + sentence
            # Lowercasing is only for keyword matching, not the length definition.
            ws = set(tokens(text))
            mask = sum(1 << j for j, word in enumerate(q) if word in ws)
            units.append(Unit(title, sid, sentence, len(TOKEN.findall(text)), mask))
    return q, units


def static_ranking(units, budget):
    chosen = []
    for i in sorted(range(len(units)), key=lambda i: (-units[i].mask.bit_count(), units[i].length, i)):
        u = units[i]
        if u.mask and u.length <= budget:
            chosen.append(i)
            budget -= u.length
    return chosen


def greedy(units, budget, seed=()):
    chosen = list(seed)
    covered = 0
    for i in seed:
        covered |= units[i].mask
        budget -= units[i].length
    remaining = set(range(len(units))) - set(seed)
    while True:
        best = None
        key = None
        for i in remaining:
            u = units[i]
            gain = (u.mask & ~covered).bit_count()
            if u.length > budget or not gain:
                continue
            # Zero-word positive-mask units (unusual Unicode case) are free.
            score = (Fraction(gain, u.length) if u.length else float("inf"), -u.length, -i)
            if key is None or score > key:
                best, key = i, score
        if best is None:
            return chosen
        chosen.append(best)
        remaining.remove(best)
        covered |= units[best].mask
        budget -= units[best].length


def dynamic_programming(units, budget):
    # mask -> (minimum length, subset bitset). The latter is a recoverable path.
    # Smaller numeric subset bitset wins a length tie: prefer excluding the
    # latest differing original unit. This order is preserved by future choices.
    dp = {0: (0, 0)}
    for i, u in enumerate(units):
        nxt = dp.copy()  # skip; transitions read only the previous layer
        for mask, (cost, subset) in dp.items():
            new_cost = cost + u.length
            if new_cost > budget:
                continue
            new_mask = mask | u.mask
            candidate = (new_cost, subset | (1 << i))
            if new_mask not in nxt or candidate < nxt[new_mask]:
                nxt[new_mask] = candidate
        dp = nxt
    mask, (cost, subset) = min(dp.items(), key=lambda x: (-x[0].bit_count(), x[1][0], x[1][1]))
    return [i for i in range(len(units)) if subset & (1 << i)]


def objective(units, chosen):
    mask = 0
    cost = 0
    for i in chosen:
        mask |= units[i].mask
        cost += units[i].length
    return mask.bit_count(), cost


def exhaustive(units, budget):
    if len(units) > 12:
        raise ValueError('Enumeration restricted to at most 12 units')
    best = None
    for bits in range(1 << len(units)):
        s = [i for i in range(len(units)) if bits & (1 << i)]
        cov, length = objective(units, s)
        if length <= budget:
            key = (-cov, length, bits)
            if best is None or key < best:
                best = key
    return [i for i in range(len(units)) if best[2] & (1 << i)]


def bridge_select(units, budget):
    """Fixed two-round heuristic. No answer or supporting_facts input."""
    seed = greedy(units, budget // 2)
    chosen = seed.copy()
    used_titles = {units[i].title for i in seed}
    remaining_budget = budget - objective(units, chosen)[1]
    # Match whole normalized title phrases in the seed sentence BODY.
    bodies = [' ' + ' '.join(tokens(units[i].sentence)) + ' ' for i in seed]
    links = {}
    for i, u in enumerate(units):
        if u.title in used_titles or u.title in links:
            continue
        title_words = tokens(re.sub(r'\s*\([^)]*\)\s*$', '', u.title))
        if not any(w not in STOP and len(w) >= 3 for w in title_words):
            continue
        phrase = ' ' + ' '.join(title_words) + ' '
        count = sum(phrase in body for body in bodies)
        if count:
            links[u.title] = (count, i)
    trace = []
    for title in sorted(links, key=lambda t: (-links[t][0], links[t][1])):
        feasible = [i for i, u in enumerate(units) if u.title == title and u.length <= remaining_budget]
        if not feasible:
            continue
        # Prefer the lead/earliest feasible sentence, even if its Q mask is zero.
        i = min(feasible, key=lambda i: (units[i].sentence_id, units[i].length, i))
        chosen.append(i)
        remaining_budget -= units[i].length
        trace.append({'title': title, 'seed_mentions': links[title][0], 'unit_index': i})
    return greedy(units, budget, chosen), {'seed': seed, 'links': trace}


METHODS = {'Static Ranking': static_ranking, 'Greedy': greedy, 'Dynamic Programming': dynamic_programming}


def evaluate(units, chosen, gold):
    coverage, words = objective(units, chosen)
    selected = {(units[i].title, units[i].sentence_id) for i in chosen}
    return dict(coverage=coverage, words=words, recall=len(selected & gold) / len(gold), full=int(gold <= selected))


def describe(units, ids, q, gold):
    return [{'index': i, 'title': units[i].title, 'sentence_id': units[i].sentence_id,
             'sentence': units[i].sentence, 'words': units[i].length,
             'keywords': [w for j, w in enumerate(q) if units[i].mask & (1 << j)],
             'is_support': (units[i].title, units[i].sentence_id) in gold} for i in ids]


def self_test():
    import random
    # Basic tie and a strict greedy counterexample.
    us = [Unit('', i, '', length, mask) for i, (length, mask) in enumerate([(3, 3), (4, 12), (4, 48)])]
    assert objective(us, greedy(us, 8)) == (4, 7)
    assert objective(us, dynamic_programming(us, 8)) == (4, 7)  # basic tie
    # Separate strict failure: overlap makes locally attractive first choice bad.
    us = [Unit('', i, '', length, mask) for i, (length, mask) in enumerate([(3, 3), (4, 5), (4, 10)])]
    assert objective(us, greedy(us, 8))[0] == 3
    assert objective(us, dynamic_programming(us, 8))[0] == 4
    rng = random.Random(42)
    for _ in range(100):
        us = [Unit('', i, '', rng.randint(1, 15), rng.randrange(64)) for i in range(rng.randint(0, 12))]
        b = rng.randint(0, 40)
        assert dynamic_programming(us, b) == exhaustive(us, b)
        assert objective(us, greedy(us, b))[0] <= objective(us, greedy(us, b + 1))[0]
    print('Self-test passed: 100 randomized enumeration comparisons + greedy counterexample.')


def run(path, out, repeats, allow_first100=False):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    if digest == OFFICIAL_SHA256 and len(data) == 7405:
        source = 'verified original'
    elif allow_first100 and len(data) == 100:
        source = 'user-extracted first 100; original hash not verified'
    else:
        raise ValueError('Use verified original file or explicitly pass --allow-first100 for a 100-question extract.')
    assert len({r['_id'] for r in data[:100]}) == 100
    rows = data[:100]
    out.mkdir(parents=True, exist_ok=True)
    prepared = [prepare(r['question'], r['context']) for r in rows]
    # Selection stage: only question/context-derived units enter these functions.
    selected = {}
    times = {}
    improved = {}
    for index, (q, units) in enumerate(prepared):
        for b in BUDGETS:
            for name, algorithm in METHODS.items():
                elapsed = []
                for _ in range(repeats):
                    t0 = perf_counter_ns()
                    s = algorithm(units, b)
                    elapsed.append((perf_counter_ns() - t0) / 1e6)
                assert len(set(s)) == len(s) and objective(units, s)[1] <= b
                selected[index, b, name] = s
                times[index, b, name] = median(elapsed)
            best = objective(units, selected[index, b, 'Dynamic Programming'])[0]
            assert all(best >= objective(units, selected[index, b, k])[0] for k in METHODS)
        s, trace = bridge_select(units, 120)
        assert len(set(s)) == len(s) and objective(units, s)[1] <= 120
        improved[index] = (s, trace)
    # Verify first question's first 12 original sentence units (not ranked).
    q0, u0 = prepared[0]
    small = u0[:12]
    validation = []
    for b in BUDGETS:
        d, e = dynamic_programming(small, b), exhaustive(small, b)
        assert d == e
        validation.append(dict(budget=b, coverage=objective(small, d)[0], words=objective(small, d)[1], chosen=d))
    # Evaluation stage begins only after ALL selection/parameters are fixed.
    details = []
    metrics = []
    for index, (r, (q, units)) in enumerate(zip(rows, prepared)):
        gold = {tuple(x) for x in r['supporting_facts']}
        assert gold and gold <= {(u.title, u.sentence_id) for u in units}
        rec = dict(index=index+1, _id=r['_id'], question=r['question'], keywords=q,
                   gold=describe(units, [i for i,u in enumerate(units) if (u.title,u.sentence_id) in gold], q, gold), results={})
        for b in BUDGETS:
            rec['results'][str(b)] = {}
            for name in METHODS:
                s = selected[index,b,name]
                result = evaluate(units,s,gold)
                result['ms'] = times[index,b,name]
                metrics.append(dict(_id=r['_id'], budget=b, method=name, **result))
                rec['results'][str(b)][name] = dict(**result, selected=describe(units,s,q,gold))
        s, trace = improved[index]
        rec['bridge'] = dict(**evaluate(units,s,gold), selected=describe(units,s,q,gold), trace=trace)
        details.append(rec)
    summary = []
    for b in BUDGETS:
        for method in METHODS:
            group = [r for r in metrics if r['budget']==b and r['method']==method]
            summary.append(dict(budget=b, method=method, **{k: mean(r[k] for r in group) for k in ('coverage','words','recall','full','ms')}, total_ms=sum(r['ms'] for r in group)))
    changes = {}
    for method in METHODS:
        changes[method] = {}
        for a,b in zip(BUDGETS,BUDGETS[1:]):
            changes[method][f'{a}->{b}'] = {k:sum(r['results'][str(b)][method][k] < r['results'][str(a)][method][k] for r in details) for k in ('coverage','recall')}
    # Post-hoc case choice, never passed back to selection algorithms.
    case1 = max(details, key=lambda r: (r['results']['120']['Dynamic Programming']['coverage'] - r['results']['120']['Greedy']['coverage'], -r['index']))
    case2 = max((r for r in details if r is not case1), key=lambda r: (max(r['results']['120'][k]['recall'] for k in ('Greedy','Static Ranking')) - r['results']['120']['Dynamic Programming']['recall'], -r['index']))
    case3 = max(details, key=lambda r: (r['bridge']['recall'] - r['results']['120']['Greedy']['recall'], len(r['bridge']['trace']['links']), -r['index']))
    report = dict(file=path.name, source=source, sha256=digest, sample_count=len(rows), ids=[r['_id'] for r in rows],
                  environment=dict(python=platform.python_version(), platform=platform.platform(), timing_repeats=repeats),
                  validation=dict(_id=rows[0]['_id'], keywords=q0, units=describe(small,range(len(small)),q0,set()), results=validation),
                  summary=summary, decreases=changes, cases=[case1,case2], bridge_case=case3,
                  bridge_summary={k:mean(r['bridge'][k] for r in details) for k in ('coverage','words','recall','full')})
    for filename, obj in [('report_data.json',report),('details.json',details)]:
        (out/filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'sample_ids.txt').write_text('\n'.join(report['ids'])+'\n',encoding='utf-8')
    for filename, records in [('summary.csv',summary),('per_question.csv',metrics)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Results written to', out)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('data',type=Path,nargs='?')
    p.add_argument('--out',type=Path,default=Path('results'))
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--allow-first100',action='store_true')
    args=p.parse_args()
    if args.self_test: self_test()
    if args.data:
        if args.repeats < 1: p.error('--repeats must be positive')
        run(args.data,args.out,args.repeats,args.allow_first100)
    elif not args.self_test: p.error('Provide original JSON or --self-test')
