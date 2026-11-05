"""Genera notebooks/analisis_enlaces_wikipedia.ipynb."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Análisis de enlaces de Wikipedia (inglés): PageRank, HITS y PageRank sensible a tópico

**Datos:** `enwiki-latest-pages-articles-multistream.xml` (~118 GB sin comprimir).

**Pipeline** (los pasos pesados se ejecutan como scripts fuera del notebook):
1. `src/extract_links.py` — parseo en streaming y en paralelo del XML (rangos de bytes, 12 procesos). Solo artículos (`ns=0`): títulos, redirecciones, enlaces `[[...]]` y categorías → Parquet.
2. `src/build_graph.py` — resolución de redirecciones, mapeo título→índice, deduplicación → lista de aristas `src → dst`.
3. **Este notebook** — matrices dispersas CSR (SciPy) e implementación propia de los algoritmos por iteración de potencia.

> Con 16 GB de RAM y millones de nodos no es viable NetworkX sobre el grafo completo; se usa solo para validar las implementaciones en un subgrafo pequeño."""),
code("""import time, re
import numpy as np, scipy.sparse as sp, polars as pl
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import spearmanr, kendalltau
sns.set_theme(style="whitegrid")
pl.Config.set_tbl_rows(20); pl.Config.set_fmt_str_lengths(60)

GRAPH_DIR = "../data/graph"
BETA, TOL, MAX_IT = 0.85, 1e-9, 200"""),
md("## 1. Carga del grafo"),
code("""t = time.time()
nodes = pl.read_parquet(f"{GRAPH_DIR}/nodes.parquet")
E = pl.read_parquet(f"{GRAPH_DIR}/edges.parquet")
src, dst = E["src"].to_numpy().astype(np.int32), E["dst"].to_numpy().astype(np.int32)
del E
N, M = nodes.height, len(src)
# A[i, j] = 1 si i enlaza a j
A = sp.csr_array((np.ones(M, np.float32), (src, dst)), shape=(N, N))
AT = A.T.tocsr()
del src, dst
out_deg = np.asarray(A.sum(axis=1)).ravel()
in_deg = np.asarray(AT.sum(axis=1)).ravel()
titles = nodes["title"].to_numpy()
print(f"N = {N:,} artículos | M = {M:,} enlaces | {time.time()-t:.0f}s")
print(f"grado medio = {M/N:.1f} | nodos colgantes (sin salida) = {(out_deg==0).sum():,} | sin entrada = {(in_deg==0).sum():,}")"""),
code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, d, name in [(axes[0], in_deg, "entrada"), (axes[1], out_deg, "salida")]:
    vals, cnt = np.unique(d[d > 0].astype(np.int64), return_counts=True)
    ax.loglog(vals, cnt, ".", ms=3, alpha=.6)
    ax.set(title=f"Distribución de grado de {name}", xlabel="grado k", ylabel="# artículos")
plt.tight_layout(); plt.show()

def top(scores, k=20, label="score"):
    idx = np.argsort(-scores)[:k]
    return pl.DataFrame({"rank": np.arange(1, k+1), "título": titles[idx], label: scores[idx]})

top(in_deg, label="grado_entrada")"""),
md(r"""## 2. PageRank

Iteración de potencia sobre la matriz estocástica por columnas $M_{ji} = 1/\\text{out}(i)$ si $i\\to j$:

$$r^{(t+1)} = \\beta\\, M r^{(t)} + \\big(\\beta\\, d^\\top r^{(t)} + 1-\\beta\\big)\\, v$$

donde $d$ marca los nodos colgantes (su masa se redistribuye según $v$) y $v$ es el vector de teletransporte. Para PageRank clásico $v = \\mathbf{1}/N$; para el sensible a tópico $v$ es uniforme sobre el conjunto del tópico $S$ (así ambos casos usan la misma función)."""),
code("""dangling = out_deg == 0
inv_out = np.where(dangling, 0, 1.0 / np.maximum(out_deg, 1)).astype(np.float64)

def pagerank(v=None, beta=BETA, tol=TOL, max_it=MAX_IT, verbose=True):
    v = np.full(N, 1.0 / N) if v is None else v / v.sum()
    r = v.copy()
    for it in range(1, max_it + 1):
        r_new = beta * (AT @ (r * inv_out))
        r_new += (beta * r[dangling].sum() + 1 - beta) * v
        err = np.abs(r_new - r).sum()
        r = r_new
        if err < tol:
            break
    if verbose: print(f"convergió en {it} iteraciones (L1 = {err:.2e}), suma = {r.sum():.6f}")
    return r

t = time.time(); pr = pagerank(); print(f"{time.time()-t:.0f}s")
top(pr, 25, "pagerank")"""),
md(r"""## 3. HITS (hubs y autoridades)

$a \\leftarrow A^\\top h$, $h \\leftarrow A a$, normalizando en cada paso. Converge a los vectores propios principales de $A^\\top A$ (autoridades) y $A A^\\top$ (hubs).

Lo aplicamos de dos formas: **(a)** global sobre todo el grafo y **(b)** en su forma original de Kleinberg, dependiente de una consulta: conjunto raíz + vecinos (conjunto base)."""),
code("""def hits(A, AT, tol=1e-8, max_it=MAX_IT, verbose=True):
    n = A.shape[0]
    h = np.full(n, 1 / np.sqrt(n))
    for it in range(1, max_it + 1):
        a = AT @ h; a /= np.linalg.norm(a)
        h_new = A @ a; h_new /= np.linalg.norm(h_new)
        err = np.abs(h_new - h).sum(); h = h_new
        if err < tol: break
    if verbose: print(f"HITS convergió en {it} iteraciones (L1 = {err:.2e})")
    return h, a

t = time.time(); hub, auth = hits(A, AT); print(f"{time.time()-t:.0f}s")
display(top(auth, 20, "autoridad")); display(top(hub, 20, "hub"))"""),
md("""Observación esperada: el HITS global suele quedar dominado por una comunidad densa (*tightly-knit community effect*) — las autoridades/hubs top no son necesariamente las páginas “más importantes”, sino las del núcleo más densamente interconectado (p.ej. listas y plantillas de navegación incluidas en el texto)."""),
md(r"""## 4. PageRank sensible a tópico

Cada tópico se define por un conjunto de teletransporte $S$: artículos cuyas categorías coinciden con una expresión regular. $v_i = 1/|S|$ si $i\\in S$, 0 si no."""),
code("""cats = pl.read_parquet(f"{GRAPH_DIR}/cats.parquet")
print(f"{cats.height:,} pares (artículo, categoría), {cats['category'].n_unique():,} categorías distintas")

TOPICS = {
    "Matemáticas":  r"(?i)\\bmathemat|\\btheorems?\\b|\\balgebra|\\bgeometry\\b|number theory",
    "Física":       r"(?i)\\bphysics\\b|\\bphysicists\\b|quantum|thermodynamics",
    "Computación":  r"(?i)computer science|programming languages|\\balgorithms\\b|software engineering|artificial intelligence",
    "Música":       r"(?i)\\bmusic(al)? genres|\\bmusicians\\b|\\bsingers\\b|\\bcomposers\\b|\\bbands\\b",
    "Deportes":     r"(?i)\\bfootball\\b|\\bbasketball\\b|\\bolympic|\\bsports\\b|\\btennis\\b",
    "Biología":     r"(?i)\\bbiology\\b|\\bgenetics\\b|\\bevolution\\b|\\bbotany\\b|\\bzoology\\b|\\becology\\b",
}
topic_sets = {}
for name, rx in TOPICS.items():
    idx = cats.filter(pl.col("category").str.contains(rx))["idx"].unique().to_numpy()
    topic_sets[name] = idx
    print(f"{name:12s} |S| = {len(idx):,}")"""),
code("""tspr = {}
for name, S in topic_sets.items():
    v = np.zeros(N); v[S] = 1.0
    t = time.time(); tspr[name] = pagerank(v, verbose=False); print(f"{name}: {time.time()-t:.0f}s")

K = 15
pl.DataFrame({name: titles[np.argsort(-r)[:K]] for name, r in tspr.items()}).with_row_index("rank", 1)"""),
md("""**Combinación por consulta:** dada una consulta con probabilidades por tópico $P(\\text{tópico}\\mid q)$, el ranking es $\\sum_k P(k\\mid q)\\, r_k$ — se precalculan los vectores y se combinan en tiempo de consulta."""),
code("""def query_rank(weights, k=15):
    w = np.array([weights.get(n, 0) for n in tspr]); w = w / w.sum()
    return top(sum(wi * tspr[n] for wi, n in zip(w, tspr)), k, "score")

query_rank({"Matemáticas": 0.6, "Computación": 0.4})"""),
md("## 5. HITS dependiente de consulta (Kleinberg)\nConjunto raíz = artículos del tópico; conjunto base = raíz ∪ sucesores ∪ (hasta 50) predecesores por nodo."),
code("""rng = np.random.default_rng(0)
def base_set(root, max_pred=50):
    succ = A[root].indices if len(root) else np.array([], int)
    sub = AT[root]
    preds = [ (p if len(p) <= max_pred else rng.choice(p, max_pred, replace=False))
              for p in np.split(sub.indices, sub.indptr[1:-1]) ]
    return np.unique(np.concatenate([root, succ, *preds]))

for name in ["Matemáticas", "Música"]:
    B = base_set(topic_sets[name])
    sub = A[B][:, B].tocsr()
    h, a = hits(sub, sub.T.tocsr(), verbose=False)
    print(f"\\n{name}: conjunto base = {len(B):,} nodos, {sub.nnz:,} aristas")
    ia, ih = np.argsort(-a)[:10], np.argsort(-h)[:10]
    display(pl.DataFrame({"autoridades": titles[B[ia]], "hubs": titles[B[ih]]}).with_row_index("rank", 1))"""),
md("## 6. Comparación de rankings"),
code("""K = 1000
measures = {"grado_entrada": in_deg.astype(float), "PageRank": pr, "HITS_autoridad": auth, "HITS_hub": hub}
names = list(measures)
tops = {n: set(np.argsort(-s)[:K]) for n, s in measures.items()}
jac = np.array([[len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for b in names] for a in names])
# correlación de rangos sobre la unión de los top-K
U = np.array(sorted(set().union(*tops.values())))
rho = np.array([[spearmanr(measures[a][U], measures[b][U])[0] for b in names] for a in names])

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.heatmap(jac, annot=True, fmt=".2f", xticklabels=names, yticklabels=names, cmap="Blues", ax=ax[0], vmin=0, vmax=1)
ax[0].set_title(f"Jaccard de los top-{K}")
sns.heatmap(rho, annot=True, fmt=".2f", xticklabels=names, yticklabels=names, cmap="RdBu_r", ax=ax[1], vmin=-1, vmax=1)
ax[1].set_title(f"Spearman ρ sobre la unión de top-{K}")
plt.tight_layout(); plt.show()"""),
code("""fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].loglog(in_deg + 1, pr, ",", alpha=.3); ax[0].set(xlabel="grado de entrada + 1", ylabel="PageRank", title="PageRank vs grado de entrada")
ax[1].loglog(pr, auth + 1e-20, ",", alpha=.3); ax[1].set(xlabel="PageRank", ylabel="autoridad HITS", title="PageRank vs autoridad HITS")
plt.tight_layout(); plt.show()

# Cuánto se mueve cada tópico respecto al PageRank general (top-100)
gen = set(np.argsort(-pr)[:100])
print({n: len(gen & set(np.argsort(-r)[:100])) for n, r in tspr.items()}, "← artículos compartidos con el top-100 general")"""),
code("""# Sensibilidad a β (factor de amortiguamiento)
res = {}
for b in [0.5, 0.7, 0.85, 0.95]:
    t = time.time(); r = pagerank(beta=b, verbose=False)
    res[b] = r; print(f"β={b}: {time.time()-t:.0f}s")
ref = set(np.argsort(-res[0.85])[:100])
print({b: len(ref & set(np.argsort(-r)[:100])) for b, r in res.items()}, "← solapamiento top-100 con β=0.85")"""),
md("## 7. Validación contra NetworkX (subgrafo pequeño)"),
code("""import networkx as nx
S = np.argsort(-in_deg)[:3000]
sub = A[S][:, S].tocoo()
G = nx.DiGraph(); G.add_nodes_from(range(len(S))); G.add_edges_from(zip(sub.row, sub.col))
# nuestras funciones sobre el subgrafo
_A, _AT = sub.tocsr(), sub.T.tocsr()
_od = np.asarray(_A.sum(1)).ravel(); _dg = _od == 0; _io = np.where(_dg, 0, 1/np.maximum(_od, 1))
n = len(S); r = np.full(n, 1/n)
for _ in range(500):
    r = BETA * (_AT @ (r * _io)) + (BETA * r[_dg].sum() + 1 - BETA) / n
nx_pr = nx.pagerank(G, alpha=BETA, tol=1e-12, max_iter=500)
h, a = hits(_A, _AT, verbose=False)
nx_h, nx_a = nx.hits(G, max_iter=1000, tol=1e-12)
print("PageRank  max |Δ| =", np.abs(r - np.array([nx_pr[i] for i in range(n)])).max())
print("Autoridad Spearman ρ =", spearmanr(a, [nx_a[i] for i in range(n)])[0])"""),
md("""## 8. Conclusiones

- **PageRank** premia artículos citados por muchos artículos *importantes*; su top coincide en gran parte con el grado de entrada pero corrige páginas infladas por plantillas/listas.
- **HITS global** se concentra en el núcleo más denso del grafo (efecto de comunidad), por eso difiere mucho de PageRank; su versión dependiente de consulta da resultados más interpretables (autoridades = artículos centrales del tema, hubs = listas/índices).
- **PageRank sensible a tópico** mantiene la robustez de PageRank pero sesga el teletransporte, mostrando los artículos centrales de cada dominio; los vectores precalculados se combinan linealmente por consulta.
- Escalabilidad: todo cabe en memoria como CSR (`int32` + `float32`); cada iteración es un producto matriz-vector disperso, O(M)."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "p3", "display_name": "Python (p3)", "language": "python"}})
nbf.write(nb, "notebooks/analisis_enlaces_wikipedia.ipynb")
print("ok")
