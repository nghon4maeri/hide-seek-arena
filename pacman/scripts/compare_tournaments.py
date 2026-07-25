import json

r1 = json.load(open(r'D:\\Giselle_\\My Project\\hide-seek-arena\\pacman\\submission\\tournament_round1.json', encoding='utf-8'))
r2 = json.load(open(r'D:\\Giselle_\\My Project\\hide-seek-arena\\pacman\\submission\\tournament_round2_full.json', encoding='utf-8'))

teams = sorted(set(r1['results'].keys()) & set(r2['results'].keys()))
print(f'{"Opp":<8} | {"R1 Pac%":<8} | {"R2 Pac%":<8} | dPac | {"R1 Ghost%":<10} | {"R2 Ghost%":<10} | dGhost')
print('-'*92)
p1_tot, p2_tot, p1_n, p2_n = 0, 0, 0, 0
g1_tot, g2_tot, g1_n, g2_n = 0, 0, 0, 0
for t in teams:
    s1p = r1['results'][t]['as_pacman']['summary']
    s2p = r2['results'][t]['as_pacman']['summary']
    s1g = r1['results'][t]['as_ghost']['summary']
    s2g = r2['results'][t]['as_ghost']['summary']
    p1 = s1p['24127457_win_rate']
    p2 = s2p['24127457_win_rate']
    g1 = s1g['24127457_win_rate']
    g2 = s2g['24127457_win_rate']
    p1_tot += s1p['24127457_wins']; p1_n += s1p['completed']
    p2_tot += s2p['24127457_wins']; p2_n += s2p['completed']
    g1_tot += s1g['24127457_wins']; g1_n += s1g['completed']
    g2_tot += s2g['24127457_wins']; g2_n += s2g['completed']
    dp = f'{p2-p1:+.0f}'
    dg = f'{g2-g1:+.0f}'
    print(f'{t:<8} | {p1:>5.0f}%   | {p2:>5.0f}%   | {dp:>4} | {g1:>7.0f}%   | {g2:>7.0f}%   | {dg:>5}')
print('-'*92)
print(f'{"TOTAL":<8} | {100*p1_tot/p1_n:>5.1f}%   | {100*p2_tot/p2_n:>5.1f}%   | {100*(p2_tot/p2_n - p1_tot/p1_n):+.1f}  | {100*g1_tot/g1_n:>7.1f}%   | {100*g2_tot/g2_n:>7.1f}%   | {100*(g2_tot/g2_n - g1_tot/g1_n):+.1f}')
print()
print(f'Pacman R1: {p1_tot}/{p1_n} ({100*p1_tot/p1_n:.1f}%)')
print(f'Pacman R2: {p2_tot}/{p2_n} ({100*p2_tot/p2_n:.1f}%)')
print(f'Ghost  R1: {g1_tot}/{g1_n} ({100*g1_tot/g1_n:.1f}%)')
print(f'Ghost  R2: {g2_tot}/{g2_n} ({100*g2_tot/g2_n:.1f}%)')
