"""Did the extra exposure land where the schedule aimed it?

    python3 tools/compare_runs.py run_01_failures.json run_02_failures.json

The v0.1.1 schedule writes a fact three times when its subject is two tokens or
shorter, twice at three, once above. If that is why accuracy rose, the gain has
to sit in the bands that were written more often. A gain spread evenly across
every band would mean the longer run did the work and the schedule did not, and
the two are worth telling apart before spending another run on the same idea.
"""
import collections
import json
import sys


def by_child(path):
    d = json.load(open(path, encoding="utf-8"))
    return d, {r["child"]: r for r in d["rows"]}


def repeats(name_tokens):
    """What the v0.1.1 schedule gave this name. Mirrors src/phrasings.py."""
    if name_tokens <= 2:
        return 3
    if name_tokens == 3:
        return 2
    return 1


def main():
    d1, a = by_child(sys.argv[1])
    d2, b = by_child(sys.argv[2])
    shared = sorted(set(a) & set(b))
    print(f"{len(shared)} facts in both runs\n")

    n1 = sum(a[c]["ok"] for c in shared)
    n2 = sum(b[c]["ok"] for c in shared)
    print(f"overall   {n1}/{len(shared)} {n1/len(shared):7.1%}"
          f"  ->  {n2}/{len(shared)} {n2/len(shared):7.1%}\n")

    per = collections.defaultdict(lambda: [0, 0, 0])
    for c in shared:
        t = a[c]["name_tokens"]
        per[t][0] += 1
        per[t][1] += a[c]["ok"]
        per[t][2] += b[c]["ok"]
    print(f"{'name tokens':>11} {'said':>5} {'n':>5} {'run 1':>8} {'run 2':>8} {'diff':>8}")
    for t in sorted(per):
        n, h1, h2 = per[t]
        print(f"{t:>11} {repeats(t):>4}x {n:>5} {h1/n:8.1%} {h2/n:8.1%}"
              f" {h2/n - h1/n:+8.1%}")

    # The schedule's own claim, stated as two groups: the names it wrote more
    # often, and the names it left exactly as they were.
    print()
    for label, keep in (("written more often", lambda t: t <= 3),
                        ("left untouched", lambda t: t > 3)):
        n = sum(per[t][0] for t in per if keep(t))
        h1 = sum(per[t][1] for t in per if keep(t))
        h2 = sum(per[t][2] for t in per if keep(t))
        print(f"{label:20} {n:>5} {h1/n:8.1%} {h2/n:8.1%} {h2/n - h1/n:+8.1%}")

    fixed = [c for c in shared if not a[c]["ok"] and b[c]["ok"]]
    broke = [c for c in shared if a[c]["ok"] and not b[c]["ok"]]
    print(f"\nfixed {len(fixed)}, broke {len(broke)}")
    for label, names in (("fixed", fixed), ("broke", broke)):
        if not names:
            continue
        c = collections.Counter(b[x]["name_tokens"] for x in names)
        print(f"  {label:6} by name tokens: "
              + ", ".join(f"{t}tok {c[t]}" for t in sorted(c)))
        print(f"         e.g. {' '.join(names[:8])}")


if __name__ == "__main__":
    main()
