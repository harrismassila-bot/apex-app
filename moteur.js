/*
 * APEX - moteur de calcul côté téléphone (même méthode que moteur.py, version 1.2).
 * Entrées : le fichier forces_<code>.json publié par export.py + les matchs récents
 * que tu as saisis toi-même. Aucun chiffre n'est inventé : sans donnée, pas de marché.
 */
(function (racine) {
  "use strict";

  // ------------------------------------------------------------ outils
  const LANCZOS = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7];

  function lgamma(x) {
    if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
    x -= 1;
    let a = 0.99999999999980993;
    const t = x + 7.5;
    for (let i = 0; i < 8; i++) a += LANCZOS[i] / (x + i + 1);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  function poisson(k, lam) {
    if (lam <= 0) return k === 0 ? 1 : 0;
    return Math.exp(-lam + k * Math.log(lam) - lgamma(k + 1));
  }

  function nbPmf(k, mu, phi) {
    if (phi <= 1.05 || mu <= 0) return poisson(k, mu);
    const r = mu / (phi - 1), p = 1 / phi;
    return Math.exp(lgamma(k + r) - lgamma(r) - lgamma(k + 1) + r * Math.log(p) + k * Math.log(1 - p));
  }

  /** P(X > seuil) pour un seuil en x,5 */
  function pPlus(seuil, mu, phi) {
    let s = 0;
    for (let k = 0; k <= Math.floor(seuil); k++) s += nbPmf(k, mu, phi);
    return Math.min(1, Math.max(0, 1 - s));
  }

  function sigmoide(x) {
    if (x >= 0) return 1 / (1 + Math.exp(-x));
    const e = Math.exp(x);
    return e / (1 + e);
  }

  function logit(p) {
    p = Math.min(Math.max(p, 1e-4), 1 - 1e-4);
    return Math.log(p / (1 - p));
  }

  function jours(deIso, aIso) {
    return (Date.parse(aIso) - Date.parse(deIso)) / 864e5;
  }

  // ------------------------------------------------------------ forces
  const CLES = ["pour_dom", "contre_dom", "pour_ext", "contre_ext"];

  /**
   * Reconstruit les forces à la date `jour` : les sommes exportées sont vieillies
   * (demi-vie) puis complétées par tes matchs saisis, exactement comme moteur.py.
   */
  function construireForces(data, manuels, jour) {
    const H = data.parametres.demi_vie, K = data.parametres.prior;
    const delta = Math.pow(0.5, Math.max(0, jours(data.date_reference, jour)) / H);
    const sommes = {}, brut = {};
    for (const [s, v] of Object.entries(data.sommes_ligue)) sommes[s] = v.map(x => x * delta);
    for (const [eq, stats] of Object.entries(data.equipes)) {
      brut[eq] = {};
      for (const [s, d] of Object.entries(stats)) {
        brut[eq][s] = {};
        for (const [cle, v] of Object.entries(d)) brut[eq][s][cle] = [v[0] * delta, v[1] * delta];
      }
    }
    for (const m of manuels || []) {
      if (!(m.date < jour)) continue;
      const w = Math.pow(0.5, jours(m.date, jour) / H);
      for (const [s, hv] of Object.entries(m.v)) {
        const [h, a] = hv;
        if (h === null || a === null || h === undefined || a === undefined) continue;
        const L = sommes[s] || (sommes[s] = [0, 0, 0]);
        L[0] += w; L[1] += w * h; L[2] += w * a;
        for (const [eq, cp, cc, pour, contre] of [[m.dom, "pour_dom", "contre_dom", h, a],
                                                  [m.ext, "pour_ext", "contre_ext", a, h]]) {
          brut[eq] = brut[eq] || {};
          brut[eq][s] = brut[eq][s] || {};
          const e = brut[eq][s];
          for (const [cle, val] of [[cp, pour], [cc, contre]]) {
            const x = e[cle] || (e[cle] = [0, 0]);
            x[0] += w * val; x[1] += w;
          }
        }
      }
    }
    const F = { ligue: {}, equipes: {}, dispersion: data.dispersion || {} };
    for (const [s, [sw, sh, sa]] of Object.entries(sommes)) {
      if (sw < 10) continue;
      const mh = sh / sw, ma = sa / sw;
      F.ligue[s] = [mh, ma];
      const prior = { pour_dom: mh, contre_dom: ma, pour_ext: ma, contre_ext: mh };
      for (const [eq, stats] of Object.entries(brut)) {
        if (!stats[s]) continue;
        const f = ((F.equipes[eq] = F.equipes[eq] || {})[s] = {});
        for (const cle of CLES) {
          if (!stats[s][cle]) continue;
          const [somme, poids] = stats[s][cle];
          f[cle] = [(somme + K * prior[cle]) / (poids + K), poids];
        }
      }
    }
    return F;
  }

  function attendu(F, dom, ext, s) {
    if (!F.ligue[s]) return null;
    const [mh, ma] = F.ligue[s];
    const fd = (F.equipes[dom] || {})[s] || {}, fe = (F.equipes[ext] || {})[s] || {};
    const [pourD, nd] = fd.pour_dom || [mh, 0];
    const [contreD] = fd.contre_dom || [ma, 0];
    const [pourE, ne] = fe.pour_ext || [ma, 0];
    const [contreE] = fe.contre_ext || [mh, 0];
    return { dom: mh > 0 ? pourD * contreE / mh : 0, ext: ma > 0 ? pourE * contreD / ma : 0, n: Math.min(nd, ne) };
  }

  // ------------------------------------------------------------ analyse
  /** Prépare tout ce qu'il faut pour calculer n'importe quel marché du match. */
  function analyser(data, manuels, dom, ext, jour) {
    const F = construireForces(data, manuels, jour);
    const g = attendu(F, dom, ext, "buts");
    if (!g) return null;
    let ld = g.dom, le = g.ext;
    const x = attendu(F, dom, ext, "xg");
    const PX = data.parametres.poids_xg;
    if (x && x.n >= 3 && F.ligue.xg) {
      const [gh, ga] = F.ligue.buts, [xh, xa] = F.ligue.xg;
      ld = (1 - PX) * ld + PX * x.dom * (xh > 0 ? gh / xh : 1);
      le = (1 - PX) * le + PX * x.ext * (xa > 0 ? ga / xa : 1);
    }
    const att = { buts: { dom: ld, ext: le, n: g.n } };
    for (const s of ["corners", "cartons", "cadres", "tirs"]) {
      const a = attendu(F, dom, ext, s);
      if (a) att[s] = a;
    }
    const pd = [], pe = [];
    for (let k = 0; k < 11; k++) { pd.push(poisson(k, ld)); pe.push(poisson(k, le)); }
    let p1 = 0, pn = 0;
    for (let i = 0; i < 11; i++) for (let j = 0; j < 11; j++) {
      if (i > j) p1 += pd[i] * pe[j]; else if (i === j) pn += pd[i] * pe[j];
    }
    const cal = data.calibration || {};
    const rec = (fam, p) => { const c = cal[fam] || [0, 1]; return sigmoide(c[0] + c[1] * logit(p)); };
    // 1N2 recalibré puis renormalisé (comme moteur.py)
    let r1 = rec("1N2", p1), rn = rec("1N2", pn), r2 = rec("1N2", 1 - p1 - pn);
    const t = r1 + rn + r2; r1 /= t; rn /= t; r2 /= t;
    return {
      dom, ext, attendus: att, dispersion: F.dispersion,
      prob1n2: { "1": r1, "N": rn, "2": r2 },
      probDC: { "1N": r1 + rn, "N2": rn + r2, "12": r1 + r2 },
      probBTTS: rec("Buts", (1 - pd[0]) * (1 - pe[0])),
      /** probabilité recalibrée de « plus de `ligne` » pour une famille et un côté */
      plus(fam, stat, cote, ligne) {
        const a = att[stat];
        if (!a) return null;
        const mu = cote === "total" ? a.dom + a.ext : a[cote];
        const phi = stat === "buts" ? 1 : (F.dispersion[stat] || 1);
        return rec(fam, pPlus(ligne, mu, phi));
      },
      poids: g.n,
    };
  }

  // ------------------------------------------------------------ verdict
  /**
   * Portes de décision (cahier des charges) :
   * P0 données fraîches · P2 famille calibrée / contexte · P3 valeur (EV ≥ 5 %) · P4 mise.
   */
  function verdict(o) {
    const r = { code: "NOBET", raison: "", ev: null, mise: 0 };
    if (!o.p0) { r.raison = "P0 · " + o.p0raison; return r; }
    if (o.statut === "bloque") { r.raison = "P2 · famille non calibrée (test de justesse)"; return r; }
    if (o.statut === "info") { r.code = "INFO"; r.raison = "Le marché prévoit mieux que le modèle : information seulement"; return r; }
    if (!o.cote || o.cote <= 1) { r.code = "ATTENTE"; r.raison = "Saisis la cote Betclic"; return r; }
    const ev = o.p * o.cote - 1;
    r.ev = ev;
    if (ev >= 0.05) {
      if (o.contexte) {
        if (o.p >= 0.7) { r.code = "SECURISE"; r.raison = "Valeur, mais contexte à risque : combiné seulement"; }
        else r.raison = "P2 · contexte à risque (absent, fatigue, derby…)";
      } else { r.code = "BET"; r.raison = "Valeur : edge de " + (100 * ev).toFixed(1).replace(".", ",") + " %"; }
    } else if (ev > 0 && o.p >= 0.7) {
      r.code = "SECURISE"; r.raison = "Probable mais peu de valeur : combiné seulement";
    } else {
      r.raison = "P3 · edge de " + (100 * ev).toFixed(1).replace(".", ",") + " %, sous le seuil de 5 %";
    }
    if (r.code === "BET") {
      const kelly = ev / (o.cote - 1);
      const fraction = Math.min(kelly / 4, 0.03);            // quart de Kelly, plafond 3 %
      r.mise = o.pause ? 0 : Math.floor((o.bankroll || 0) * fraction / 50) * 50;
      if (o.pause) r.raison += " · PAUSE : drawdown > 15 %";
    }
    return r;
  }

  racine.ApexMoteur = { analyser, verdict, construireForces, attendu, pPlus, poisson, lgamma };
})(typeof window !== "undefined" ? window : globalThis);
