"""
APEX - Export vers l'application (dépôt public apex-app).

Pour chaque ligue, écrit export/forces_<code>.json qui contient UNIQUEMENT :
  - les sommes pondérées par équipe et par statistique (de quoi recalculer les forces),
  - les moyennes et la dispersion de la ligue,
  - les paramètres de recalibrage du moteur (appris sur les 12 derniers mois),
  - la date du dernier match connu de chaque équipe,
  - le statut de chaque famille de marchés, fixé par les tests de justesse.
Aucun match individuel, aucune clé, aucune donnée personnelle.
"""
import json
import os
from datetime import date, datetime, timezone

import moteur

# Statut des familles, décidé par les tests de justesse V1.2 (9 octobre 2026) :
#   "candidat" = calibré : verdict BET possible (en papier tant que le feu vert n'est pas donné)
#   "info"     = calibré mais le marché prévoit mieux : jamais de BET
#   "bloque"   = pas calibré : affiché grisé, jamais de BET
STATUT = {
    "SP1": {"1N2": "info", "Double chance": "info", "Buts": "info",
            "Buts équipe": "candidat", "Corners": "candidat", "Tirs cadrés": "candidat",
            "Tirs": "candidat", "Cartons": "bloque"},
    "E0": {"1N2": "info", "Double chance": "info", "Buts": "info",
           "Buts équipe": "candidat", "Corners": "candidat", "Tirs cadrés": "candidat",
           "Tirs": "bloque", "Cartons": "bloque"},
}
# Ligues non testées : rien n'est candidat tant qu'un test de justesse ne l'a pas validé.
NON_TESTE = {f: "info" for f in ("1N2", "Double chance", "Buts", "Buts équipe",
                                  "Corners", "Tirs cadrés", "Tirs")}
NON_TESTE["Cartons"] = "bloque"


def exporter(nom, code, jour):
    matchs = moteur.charger(code)
    if not matchs:
        return None
    F = moteur.forces(matchs, jour)
    passes = [m for m in matchs if 0 < (jour - m["date"]).days <= 365]
    cal = moteur.calibrage(moteur.predire(matchs, passes))
    saison = max(m["saison"] for m in matchs)
    actives = sorted({m["dom"] for m in matchs if m["saison"] == saison}
                     | {m["ext"] for m in matchs if m["saison"] == saison})
    dernier = {}
    for m in matchs:
        for eq in (m["dom"], m["ext"]):
            dernier[eq] = m["date"].isoformat()
    return {
        "ligue": nom, "code": code, "saison": saison,
        "date_reference": jour.isoformat(),
        "genere_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dernier_match": max(m["date"] for m in matchs).isoformat(),
        "source": "football-data.co.uk, calculs APEX",
        "parametres": {"demi_vie": moteur.DEMI_VIE, "prior": moteur.PRIOR,
                       "poids_xg": moteur.POIDS_XG},
        "sommes_ligue": {s: list(v) for s, v in F["sommes"].items()},
        "dispersion": {s: v["equipe"] for s, v in F["dispersion"].items()},
        "equipes": {eq: {s: {cle: list(v) for cle, v in d.items()}
                         for s, d in F["brut"].get(eq, {}).items()}
                    for eq in actives},
        "dernier_match_equipe": {eq: dernier.get(eq) for eq in actives},
        "calibration": {fam: list(v) for fam, v in cal.items()},
        "statut": STATUT.get(code, NON_TESTE),
        "teste": code in STATUT,
    }


if __name__ == "__main__":
    jour = date.today()
    os.makedirs("export", exist_ok=True)
    lignes = []
    index = []
    for nom, code in moteur.LIGUES.items():
        d = exporter(nom, code, jour)
        if not d:
            lignes.append(f"- {nom} : pas de données")
            continue
        with open(os.path.join("export", f"forces_{code}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
        index.append({"ligue": nom, "code": code, "dernier_match": d["dernier_match"],
                      "teste": d["teste"]})
        lignes.append(f"- {nom} : {len(d['equipes'])} équipes, dernier match le {d['dernier_match']}")
    with open(os.path.join("export", "index.json"), "w", encoding="utf-8") as f:
        json.dump({"genere_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "ligues": index}, f, ensure_ascii=False)
    texte = "## Export vers l'application\n" + "\n".join(lignes)
    print(texte)
    chemin = os.environ.get("GITHUB_STEP_SUMMARY")
    if chemin:
        with open(chemin, "a", encoding="utf-8") as f:
            f.write(texte + "\n")
