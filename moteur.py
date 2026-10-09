"""
APEX - Moteur de calcul (V1).

Deux usages, choisis par la variable MODE :
  - MODE=analyse : probabilités de chaque marché pour un match (DOM contre EXT).
  - MODE=test    : test de justesse sur une saison passée (SAISON), match par match,
                   en n'utilisant que les données ANTÉRIEURES à chaque match.

Données : data/fd/<code>_<saison>.csv (football-data.co.uk), téléchargées par donnees.py.
Aucune donnée inventée : si une statistique manque, le marché correspondant est omis.

Méthode (cahier des charges, section « Comment l'appli confronte les données ») :
  1. Pour chaque statistique, chaque équipe a une force « pour » et « contre »,
     séparée domicile / extérieur, pondérée par l'ancienneté (demi-vie DEMI_VIE jours)
     et ramenée vers la moyenne de la ligue quand l'échantillon est petit (PRIOR).
  2. Valeur attendue = force pour de A × force contre de B ÷ moyenne de la ligue.
  3. Buts : loi de Poisson (xG mélangé aux buts quand il est disponible).
     Corners, cartons, tirs : loi binomiale négative (dispersion mesurée sur la ligue).
Seul la bibliothèque standard de Python est utilisée.
"""
import csv
import difflib
import json
import math
import os
import sys
from datetime import date, datetime

DOSSIER = os.path.join("data", "fd")
LIGUES = {"Liga": "SP1", "Premier League": "E0", "Serie A": "I1",
          "Bundesliga": "D1", "Ligue 1": "F1"}
DEMI_VIE = 150      # jours : un match vieux de 150 jours compte moitié moins
FENETRE = 1100      # jours : on ignore les matchs plus anciens (~3 saisons)
PRIOR = 3.0         # poids (en « matchs récents ») de la moyenne de ligue dans chaque force
POIDS_XG = 0.5      # part du xG dans l'estimation des buts, quand il existe

# statistique -> (colonne domicile, colonne extérieur)
STATS = {
    "buts": ("FTHG", "FTAG"), "xg": ("HxG", "AxG"),
    "tirs": ("HS", "AS"), "cadres": ("HST", "AST"),
    "corners": ("HC", "AC"), "fautes": ("HF", "AF"),
    "cartons": ("HY", "AY"),          # jaunes ; les rouges sont ajoutés au chargement
}


# ---------------------------------------------------------------- chargement
def nombre(v):
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def lire_date(d):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(d.strip(), fmt).date()
        except ValueError:
            continue
    return None


def charger(code):
    """Tous les matchs joués d'une ligue, toutes saisons, triés par date."""
    matchs = []
    for nom in sorted(os.listdir(DOSSIER)):
        if not (nom.startswith(code + "_") and nom.endswith(".csv")):
            continue
        saison = nom[len(code) + 1:-4]
        with open(os.path.join(DOSSIER, nom), encoding="utf-8") as f:
            for l in csv.DictReader(f):
                d = lire_date(l.get("Date", ""))
                if not d or nombre(l.get("FTHG")) is None:
                    continue
                m = {"date": d, "saison": saison, "dom": l["HomeTeam"].strip(),
                     "ext": l["AwayTeam"].strip(), "v": {}}
                for s, (ch, ca) in STATS.items():
                    h, a = nombre(l.get(ch)), nombre(l.get(ca))
                    if h is not None and a is not None:
                        m["v"][s] = (h, a)
                if "cartons" in m["v"]:  # cartons = jaunes + rouges
                    rh, ra = nombre(l.get("HR")) or 0, nombre(l.get("AR")) or 0
                    h, a = m["v"]["cartons"]
                    m["v"]["cartons"] = (h + rh, a + ra)
                # cotes de clôture (Pinnacle, sinon moyenne du marché)
                for pre in ("PSC", "AvgC"):
                    c = [nombre(l.get(pre + x)) for x in ("H", "D", "A")]
                    if all(c):
                        m["cote_1n2"] = c
                        break
                for pre in ("PC", "AvgC"):
                    c = [nombre(l.get(pre + ">2.5")), nombre(l.get(pre + "<2.5"))]
                    if all(c):
                        m["cote_25"] = c
                        break
                matchs.append(m)
    matchs.sort(key=lambda m: m["date"])
    return matchs


# ---------------------------------------------------------------- forces
def forces(matchs, jour):
    """Forces de chaque équipe calculées avec les seuls matchs antérieurs à `jour`."""
    passe = [m for m in matchs if m["date"] < jour and (jour - m["date"]).days <= FENETRE]
    res = {"ligue": {}, "equipes": {}, "dispersion": {}, "nb": len(passe),
           "sommes": {}, "brut": {}}   # sommes pondérées brutes, pour l'export vers l'appli
    for s in STATS:
        sw = sh = sa = 0.0
        acc = {}
        for m in passe:
            if s not in m["v"]:
                continue
            h, a = m["v"][s]
            w = 0.5 ** ((jour - m["date"]).days / DEMI_VIE)
            sw += w; sh += w * h; sa += w * a
            for eq, cle_pour, cle_contre, pour, contre in (
                    (m["dom"], "pour_dom", "contre_dom", h, a),
                    (m["ext"], "pour_ext", "contre_ext", a, h)):
                e = acc.setdefault(eq, {})
                for cle, val in ((cle_pour, pour), (cle_contre, contre)):
                    x = e.setdefault(cle, [0.0, 0.0, 0.0])
                    x[0] += w * val; x[1] += w; x[2] += w * val * val
        if sw < 10:          # pas assez de matchs dans la ligue pour cette stat
            continue
        mh, ma = sh / sw, sa / sw
        res["ligue"][s] = (mh, ma)
        res["sommes"][s] = (sw, sh, sa)
        # Dispersion (variance / moyenne) mesurée À L'INTÉRIEUR de chaque équipe
        # (même équipe, même lieu) : on retire ainsi l'écart de niveau entre équipes,
        # qui gonflait artificiellement la dispersion et aplatissait les probabilités.
        num = den = moy = 0.0
        for e in acc.values():
            for cle in ("pour_dom", "pour_ext"):
                if cle in e and e[cle][1] > 1:
                    sx, sw_, sxx = e[cle]
                    num += sxx - sx * sx / sw_
                    den += sw_
                    moy += sx
        phi_eq = max(1.0, (num / den) / (moy / den)) if den > 0 and moy > 0 else 1.0
        # somme de deux comptages indépendants de même dispersion : même rapport variance/moyenne
        res["dispersion"][s] = {"total": phi_eq, "equipe": phi_eq}
        prior = {"pour_dom": mh, "contre_dom": ma, "pour_ext": ma, "contre_ext": mh}
        for eq, e in acc.items():
            f = res["equipes"].setdefault(eq, {})
            f[s] = {}
            b = res["brut"].setdefault(eq, {}).setdefault(s, {})
            for cle, (somme, poids, _) in e.items():
                f[s][cle] = ((somme + PRIOR * prior[cle]) / (poids + PRIOR), poids)
                b[cle] = (somme, poids)
    return res


def attendu(F, dom, ext, s):
    """Valeurs attendues (domicile, extérieur) de la statistique s, et poids des données."""
    if s not in F["ligue"]:
        return None
    mh, ma = F["ligue"][s]
    fd = F["equipes"].get(dom, {}).get(s, {})
    fe = F["equipes"].get(ext, {}).get(s, {})
    pour_d, nd = fd.get("pour_dom", (mh, 0))
    contre_d, _ = fd.get("contre_dom", (ma, 0))
    pour_e, ne = fe.get("pour_ext", (ma, 0))
    contre_e, _ = fe.get("contre_ext", (mh, 0))
    lam_d = pour_d * contre_e / mh if mh > 0 else 0
    lam_e = pour_e * contre_d / ma if ma > 0 else 0
    return lam_d, lam_e, min(nd, ne)


# ---------------------------------------------------------------- lois
def poisson(k, lam):
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if lam > 0 else float(k == 0)


def nb_pmf(k, mu, phi):
    """Binomiale négative de moyenne mu et variance phi*mu (Poisson si phi≈1)."""
    if phi <= 1.05 or mu <= 0:
        return poisson(k, mu)
    r = mu / (phi - 1)
    p = 1 / phi
    return math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                    + r * math.log(p) + k * math.log(1 - p))


def p_plus(seuil, mu, phi):
    """P(X > seuil) pour un seuil en x,5."""
    return 1 - sum(nb_pmf(k, mu, phi) for k in range(int(math.floor(seuil)) + 1))


# ---------------------------------------------------------------- marchés
def marches(F, dom, ext):
    """Dictionnaire  nom du marché -> (famille, probabilité, poids des données)."""
    out = {}
    g = attendu(F, dom, ext, "buts")
    if g is None:
        return out, {}
    ld, le, n = g
    x = attendu(F, dom, ext, "xg")
    if x and x[2] >= 3:   # xG disponible pour les deux équipes
        # Le xG de la source peut être globalement plus bas ou plus haut que les buts réels :
        # on le ramène à l'échelle des buts de la ligue avant de le mélanger.
        gh, ga = F["ligue"]["buts"]
        xh, xa = F["ligue"]["xg"]
        ld = (1 - POIDS_XG) * ld + POIDS_XG * x[0] * (gh / xh if xh > 0 else 1)
        le = (1 - POIDS_XG) * le + POIDS_XG * x[1] * (ga / xa if xa > 0 else 1)
    attendus = {"buts": (ld, le)}
    pd = [poisson(k, ld) for k in range(11)]
    pe = [poisson(k, le) for k in range(11)]
    p1 = sum(pd[i] * pe[j] for i in range(11) for j in range(11) if i > j)
    pn = sum(pd[i] * pe[i] for i in range(11))
    p2 = 1 - p1 - pn
    for nom, p in (("Victoire domicile", p1), ("Nul", pn), ("Victoire extérieur", p2)):
        out[nom] = ("1N2", p, n)
    for nom, p in (("Double chance 1N", p1 + pn), ("Double chance N2", pn + p2),
                   ("Double chance 12", p1 + p2)):
        out[nom] = ("Double chance", p, n)
    for s in (1.5, 2.5, 3.5):
        out[f"Plus de {s} buts"] = ("Buts", p_plus(s, ld + le, 1.0), n)
    out["Les deux marquent"] = ("Buts", (1 - pd[0]) * (1 - pe[0]), n)
    for qui, lam in (("domicile", ld), ("extérieur", le)):
        for s in (0.5, 1.5):
            out[f"Équipe {qui} plus de {s} buts"] = ("Buts équipe", p_plus(s, lam, 1.0), n)

    lignes = {"corners": ("Corners", (8.5, 9.5, 10.5), None),
              "cartons": ("Cartons", (3.5, 4.5, 5.5), None),
              "cadres": ("Tirs cadrés", None, (3.5, 4.5, 5.5)),
              "tirs": ("Tirs", None, (10.5, 12.5))}
    for s, (fam, seuils_total, seuils_eq) in lignes.items():
        a = attendu(F, dom, ext, s)
        if not a:
            continue
        ad, ae, ns = a
        attendus[s] = (ad, ae)
        disp = F["dispersion"][s]
        for seuil in seuils_total or ():
            out[f"Plus de {seuil} {fam.lower()} (total)"] = (fam, p_plus(seuil, ad + ae, disp["total"]), ns)
        for seuil in seuils_eq or ():
            out[f"{fam} domicile plus de {seuil}"] = (fam, p_plus(seuil, ad, disp["equipe"]), ns)
            out[f"{fam} extérieur plus de {seuil}"] = (fam, p_plus(seuil, ae, disp["equipe"]), ns)
    return out, attendus


def resultat(m, nom):
    """Vrai / faux / None (stat absente) pour un marché, d'après le match joué."""
    v = m["v"]
    gh, ga = v["buts"]
    if nom == "Victoire domicile": return gh > ga
    if nom == "Nul": return gh == ga
    if nom == "Victoire extérieur": return gh < ga
    if nom == "Double chance 1N": return gh >= ga
    if nom == "Double chance N2": return gh <= ga
    if nom == "Double chance 12": return gh != ga
    if nom == "Les deux marquent": return gh > 0 and ga > 0
    if nom.startswith("Plus de") and nom.endswith("buts"):
        return gh + ga > float(nom.split()[2])
    if nom.startswith("Équipe"):
        s = float(nom.split()[4]); return (gh if "domicile" in nom else ga) > s
    for s, fam in (("corners", "corners"), ("cartons", "cartons"),
                   ("cadres", "tirs cadrés"), ("tirs", "tirs")):
        if nom.startswith("Plus de") and nom.endswith(f"{fam} (total)"):
            if s not in v: return None
            return sum(v[s]) > float(nom.split()[2])
    for s, fam in (("cadres", "Tirs cadrés"), ("tirs", "Tirs")):
        if nom.startswith(fam + " "):
            if s not in v: return None
            reste = nom[len(fam) + 1:].split()
            if reste[0] not in ("domicile", "extérieur"): continue
            val = v[s][0] if reste[0] == "domicile" else v[s][1]
            return val > float(reste[-1])
    return None



# ---------------------------------------------------------------- recalibrage
# Le modèle brut se trompe de façon régulière sur certaines familles (trop sûr de lui,
# ou décalé). On mesure ce biais sur des matchs PASSÉS (saison précédente, ou 12 derniers
# mois pour une analyse) et on le corrige par une régression logistique à 2 paramètres
# (méthode de Platt) : p' = 1 / (1 + exp(-(a + b × logit(p)))). Jamais avec les matchs testés.
def predire(matchs, cibles):
    """Prédictions brutes pour une liste de matchs : [(match, marchés)]."""
    sortie, jour_cache, F = [], None, None
    for m in cibles:
        if m["date"] != jour_cache:
            jour_cache, F = m["date"], forces(matchs, m["date"])
        res, _ = marches(F, m["dom"], m["ext"])
        if res:
            sortie.append((m, res))
    return sortie


def sigmoide(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def ajuster(paires):
    """Paramètres (a, b) de Platt par maximum de vraisemblance (Newton). Identité si trop peu de cas."""
    if len(paires) < 200:
        return (0.0, 1.0)
    a, b = 0.0, 1.0
    for _ in range(30):
        ga = gb = haa = hab = hbb = 0.0
        for p, y in paires:
            z = logit(p)
            q = sigmoide(a + b * z)
            d = q - y
            ga += d; gb += d * z
            w = q * (1 - q)
            haa += w; hab += w * z; hbb += w * z * z
        det = haa * hbb - hab * hab
        if det <= 1e-9:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        pas = max(1.0, abs(da) + abs(db))      # pas limité : pas de divergence
        a -= da / pas; b -= db / pas
        if abs(da) + abs(db) < 1e-6:
            break
    return (max(-2.0, min(2.0, a)), max(0.2, min(3.0, b)))


def calibrage(predictions):
    """Paramètres de recalibrage par famille, appris sur des prédictions passées."""
    par_fam = {}
    for m, res in predictions:
        for nom, (fam, p, n) in res.items():
            r = resultat(m, nom)
            if r is not None:
                par_fam.setdefault(fam, []).append((p, r))
    return {fam: ajuster(lst) for fam, lst in par_fam.items()}


def recalibrer(res, cal):
    """Applique le recalibrage ; les trois issues du 1N2 sont renormalisées (somme = 100 %)."""
    out = {}
    for nom, (fam, p, n) in res.items():
        a, b = cal.get(fam, (0.0, 1.0))
        out[nom] = (fam, sigmoide(a + b * logit(p)), n)
    trio = ("Victoire domicile", "Nul", "Victoire extérieur")
    if all(k in out for k in trio):
        tot = sum(out[k][1] for k in trio)
        for k in trio:
            out[k] = (out[k][0], out[k][1] / tot, out[k][2])
        p1, pn, p2 = (out[k][1] for k in trio)
        for k, v in (("Double chance 1N", p1 + pn), ("Double chance N2", pn + p2),
                     ("Double chance 12", p1 + p2)):
            if k in out:
                out[k] = (out[k][0], v, out[k][2])
    return out

# ---------------------------------------------------------------- usages
def ecrire(texte):
    print(texte)
    chemin = os.environ.get("GITHUB_STEP_SUMMARY")
    if chemin:
        with open(chemin, "a", encoding="utf-8") as f:
            f.write(texte + "\n")


def trouver(nom, equipes):
    if nom in equipes:
        return nom
    proche = difflib.get_close_matches(nom, equipes, n=1, cutoff=0.6)
    if proche:
        return proche[0]
    sys.exit(f"Équipe introuvable : « {nom} ». Noms disponibles : {', '.join(sorted(equipes))}")


def analyse(code, nom_ligue, dom, ext, jour):
    matchs = charger(code)
    if not matchs:
        sys.exit("Aucune donnée : lance d'abord « Données matchs ».")
    recentes = {m["dom"] for m in matchs[-200:]} | {m["ext"] for m in matchs[-200:]}
    dom, ext = trouver(dom, sorted(recentes)), trouver(ext, sorted(recentes))
    F = forces(matchs, jour)
    res, att = marches(F, dom, ext)
    passes = [m for m in matchs if 0 < (jour - m["date"]).days <= 365]
    cal = calibrage(predire(matchs, passes))
    res = recalibrer(res, cal)
    dernier = max(m["date"] for m in matchs)
    retard = (jour - dernier).days
    t = [f"## {dom} – {ext} · {nom_ligue} · analyse du {jour.isoformat()}", ""]
    t.append(f"Données : {F['nb']} matchs (source football-data.co.uk), "
             f"dernier match enregistré le {dernier.isoformat()} ({retard} jours).")
    if retard > 14:
        t.append(f"> ⚠️ **Données anciennes de {retard} jours** : les matchs récents manquent. "
                 "Résultat indicatif seulement (porte P0 non franchie).")
    t += ["", "| Valeurs attendues | " + dom + " | " + ext + " |", "| --- | --- | --- |"]
    noms = {"buts": "Buts", "corners": "Corners", "cartons": "Cartons",
            "cadres": "Tirs cadrés", "tirs": "Tirs"}
    for s, (a, b) in att.items():
        t.append(f"| {noms[s]} | {a:.2f} | {b:.2f} |")
    t += ["", "| Marché | Probabilité | Cote mini rentable | Poids des données |",
          "| --- | --- | --- | --- |"]
    for nom, (fam, p, n) in sorted(res.items(), key=lambda x: -x[1][1]):
        cote = f"{1 / p:.2f}" if p > 0.01 else "—"
        t.append(f"| {nom} | {100 * p:.1f} % | {cote} | {n:.1f} |")
    t += ["", "_Cote mini rentable = 1 ÷ probabilité : en dessous, le pari perd de l'argent "
          "sur la durée. Aucun verdict BET tant que le test de justesse n'a pas validé "
          "la famille de marchés._"]
    ecrire("\n".join(t))


def test(code, nom_ligue, saison):
    matchs = charger(code)
    cibles = [m for m in matchs if m["saison"] == saison]
    if not cibles:
        sys.exit(f"Aucun match pour la saison {saison}.")
    saisons = sorted({m["saison"] for m in matchs})
    prec = saisons[saisons.index(saison) - 1] if saisons.index(saison) > 0 else None
    # Recalibrage « au fil de l'eau » : avant chaque mois testé, on l'apprend sur les
    # 12 mois précédents (fin de saison précédente + début de saison), jamais sur l'avenir.
    avant = predire(matchs, [m for m in matchs if m["saison"] == prec]) if prec else []
    testes = predire(matchs, cibles)
    historique = avant + testes
    familles = {}           # famille -> liste (p brut, p recalibré, issue)
    marche_1n2 = []         # (p modèle, p marché, issue, cote) pour le 1N2
    marche_25 = []
    mois_cache, cal = None, {}
    for m, brut in testes:
        mois = (m["date"].year, m["date"].month)
        if mois != mois_cache:
            debut = date(*mois, 1)
            fenetre = [(x, r) for x, r in historique if 0 < (debut - x["date"]).days <= 365]
            mois_cache, cal = mois, calibrage(fenetre)
        res = recalibrer(brut, cal)
        for nom, (fam, p, n) in res.items():
            r = resultat(m, nom)
            if r is not None:
                familles.setdefault(fam, []).append((brut[nom][1], p, r))
        if "cote_1n2" in m:
            c = m["cote_1n2"]; inv = [1 / x for x in c]; s = sum(inv)
            gh, ga = m["v"]["buts"]
            issue = 0 if gh > ga else (1 if gh == ga else 2)
            pm = [res["Victoire domicile"][1], res["Nul"][1], res["Victoire extérieur"][1]]
            marche_1n2.append((pm, [x / s for x in inv], issue, c))
        if "cote_25" in m:
            c = m["cote_25"]; inv = [1 / x for x in c]; s = sum(inv)
            issue = 0 if sum(m["v"]["buts"]) > 2.5 else 1
            p = res["Plus de 2.5 buts"][1]
            marche_25.append(([p, 1 - p], [x / s for x in inv], issue, c))

    t = [f"## Test de justesse · {nom_ligue} {saison[:2]}-{saison[2:]} · {len(cibles)} matchs", "",
         "Chaque match est prédit avec les seules données antérieures à sa date. "
         + ("Le recalibrage de chaque mois est appris sur les 12 mois précédents uniquement."
            if prec else "Pas de saison précédente : recalibrage limité."), "",
         "### 1. Calibration par famille de marchés",
         "Critère : dans chaque tranche d'au moins 50 cas, l'écart entre la probabilité "
         "annoncée et la fréquence réelle doit rester ≤ 5 points (tolérance élargie au "
         "hasard statistique pour les petites tranches), et le modèle doit battre la référence.", "",
         "| Famille | Cas | Brier brut | Brier recalibré | Brier référence | Pire écart corrigé | Verdict |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    rapport = {"ligue": nom_ligue, "saison": saison, "familles": {}}
    for fam, lst in sorted(familles.items()):
        n = len(lst)
        brier_brut = sum((pb - r) ** 2 for pb, _, r in lst) / n
        lst = [(p, r) for _, p, r in lst]
        freq = sum(r for _, r in lst) / n
        brier = sum((p - r) ** 2 for p, r in lst) / n
        ref = sum((freq - r) ** 2 for _, r in lst) / n
        tranches = []
        pire = 0.0
        for b in range(10):
            dans = [(p, r) for p, r in lst if b / 10 <= p < (b + 1) / 10 or (b == 9 and p == 1)]
            if not dans:
                continue
            pm = sum(p for p, _ in dans) / len(dans)
            fr = sum(r for _, r in dans) / len(dans)
            tranches.append({"tranche": f"{10*b}-{10*b+10} %", "cas": len(dans),
                             "annonce": round(100 * pm, 1), "reel": round(100 * fr, 1)})
            if len(dans) >= 50:
                # tolérance : 5 points, ou 2,5 écarts-types du hasard si l'échantillon est petit
                tol = max(0.05, 2.5 * math.sqrt(max(pm * (1 - pm), 1e-6) / len(dans)))
                pire = max(pire, abs(pm - fr) - tol + 0.05)
        ok = pire <= 0.05 and brier < ref
        verdict = "✅ CALIBRÉ" if ok else "❌ À REVOIR"
        t.append(f"| {fam} | {n} | {brier_brut:.4f} | {brier:.4f} | {ref:.4f} | {100 * pire:.1f} pts | {verdict} |")
        rapport["familles"][fam] = {"cas": n, "brier": brier, "brier_ref": ref,
                                    "pire_ecart": pire, "valide": ok, "tranches": tranches}
    t += ["", "_Brier : plus c'est bas, mieux c'est. « Référence » = toujours annoncer la "
          "fréquence moyenne. Un modèle utile fait mieux que la référence._"]
    a_revoir = [f for f, r in sorted(rapport["familles"].items()) if not r["valide"]]
    if a_revoir:
        t += ["", "#### Détail des familles à revoir (pour le diagnostic)", "",
              "| Famille | Tranche | Cas | Annoncé | Réel |", "| --- | --- | --- | --- | --- |"]
        for fam in a_revoir:
            for tr in rapport["familles"][fam]["tranches"]:
                if tr["cas"] >= 20:
                    t.append(f"| {fam} | {tr['tranche']} | {tr['cas']} | "
                             f"{tr['annonce']} % | {tr['reel']} % |")

    t += ["", "### 2. Face aux cotes de clôture (le test le plus dur)",
          "Paris fictifs de 1 unité, pris à la cote de clôture, quand l'edge du modèle ≥ 5 %.", "",
          "| Marché | Matchs | Perte log modèle | Perte log marché | Paris | ROI |",
          "| --- | --- | --- | --- | --- | --- |"]
    for nom, lst in (("1N2", marche_1n2), ("Plus/moins 2,5 buts", marche_25)):
        if not lst:
            t.append(f"| {nom} | 0 | — | — | — | cotes absentes |")
            continue
        ll_m = -sum(math.log(max(pm[i], 1e-9)) for pm, _, i, _ in lst) / len(lst)
        ll_b = -sum(math.log(max(pb[i], 1e-9)) for _, pb, i, _ in lst) / len(lst)
        mises = gain = 0
        for pm, _, i, c in lst:
            for k in range(len(c)):
                if pm[k] * c[k] - 1 >= 0.05:
                    mises += 1
                    gain += (c[k] - 1) if k == i else -1
        roi = f"{100 * gain / mises:+.1f} %" if mises else "—"
        t.append(f"| {nom} | {len(lst)} | {ll_m:.4f} | {ll_b:.4f} | {mises} | {roi} |")
        rapport[nom] = {"matchs": len(lst), "logloss_modele": ll_m, "logloss_marche": ll_b,
                        "paris": mises, "roi": (gain / mises if mises else None)}
    t += ["", "_Si la perte log du modèle est plus haute que celle du marché, le marché "
          "prévoit mieux que nous sur ce marché : pas de pari dessus. Un ROI sur une seule "
          "saison reste fragile ; seul le paper trading tranchera._"]
    ecrire("\n".join(t))
    os.makedirs(os.path.join("data", "rapports"), exist_ok=True)
    with open(os.path.join("data", "rapports", f"test_{code}_{saison}.json"), "w",
              encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    nom_ligue = os.environ.get("LIGUE") or "Liga"
    if nom_ligue not in LIGUES:
        sys.exit(f"Ligue inconnue : {nom_ligue}")
    code = LIGUES[nom_ligue]
    mode = (os.environ.get("MODE") or "analyse").lower()
    if mode.startswith("test"):
        test(code, nom_ligue, os.environ.get("SAISON") or "2526")
    else:
        dom, ext = os.environ.get("DOM"), os.environ.get("EXT")
        if not dom or not ext:
            sys.exit("Indique l'équipe à domicile (DOM) et à l'extérieur (EXT).")
        j = os.environ.get("DATE")
        analyse(code, nom_ligue, dom, ext, date.fromisoformat(j) if j else date.today())
