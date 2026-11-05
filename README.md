<div align="center">

# 🕸️ Análisis de Redes: PageRank, HITS y PageRank Sensible a Tópico

**Datos Masivos · Proyecto 3**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-SciPy-013243?logo=numpy&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-Parquet-CD792C?logo=polars&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-3.x-4B8BBE)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

*Tres algoritmos de análisis de enlaces aplicados a tres redes muy distintas:
colaboraciones musicales, citas científicas y la Wikipedia completa.*

</div>

---

## 📖 Descripción

Este repositorio analiza varios datasets de redes con tres algoritmos clásicos de análisis de enlaces:

| Algoritmo | Idea central |
|---|---|
| **PageRank** | Un surfista aleatorio salta por los enlaces; un nodo es importante si lo apuntan nodos importantes. |
| **HITS** | Cada nodo recibe dos puntajes: *authority* (lo apuntan buenos hubs) y *hub* (apunta a buenas authorities). |
| **PageRank sensible a tópico** | El surfista se teletransporta solo a nodos de un tema, midiendo la importancia *desde la perspectiva* de ese tópico. |

## 🗂️ Datasets

| | Dataset | Tipo de red | Nodos | Aristas | Dónde |
|---|---|---|---|---|---|
| **Parte I** | *Spotify Artist Feature Collaboration Network* (Kaggle) | No dirigida (colaboraciones) | ~156 mil artistas | ~300 mil | `analisis_pagerank_hits.ipynb` |
| **Parte II** | *High-energy physics citation network* `cit-HepPh` (SNAP) | Dirigida (citas) | ~34.5 mil artículos | ~421 mil | `analisis_pagerank_hits.ipynb` |
| **Parte III** | Dump de Wikipedia en inglés (`enwiki`) | Dirigida (hipervínculos) | ~7.25 millones de artículos | ~236 millones | `wikipedia/` |

## 📁 Estructura del repositorio

```
.
├── analisis_pagerank_hits.ipynb      # Partes I y II: Spotify + citas HEP-PH
├── wikipedia/                        # Parte III: Wikipedia a gran escala
│   ├── src/
│   │   ├── extract_links.py          #   1. XML (118 GB) → Parquet, en paralelo
│   │   ├── build_graph.py            #   2. Parquet → grafo (nodos y aristas)
│   │   └── make_notebook.py          #   3. Genera el notebook de análisis
│   └── notebooks/
│       └── analisis_enlaces_wikipedia.ipynb
├── PIPELINE.md                       # Explicación detallada del pipeline de Wikipedia
├── requirements.txt                  # Dependencias
└── README.md
```

> Los datos (`data/`) y los logs no se incluyen por su tamaño (el XML de Wikipedia pesa ~118 GB).

---

## 🎵 Parte I · Red de colaboraciones de Spotify

Los nodos son artistas y las aristas, colaboraciones (*features*). Se analiza con PageRank, HITS
(no dirigido y dirigido por popularidad) y PageRank sensible a tópico con los géneros
*reggaeton, k-pop, hip hop, pop y rock*.

**Hallazgos principales**

- Red **libre de escala**: grado promedio de 3.84 pero con artistas de hasta 1,781 colaboradores; 94.9 % en una sola componente.
- **Sesgo del dataset:** *Bach*, *Traditional* y *Sibelius* son los más centrales porque Spotify los acredita como compositores en miles de grabaciones.
- **PageRank** se correlaciona con el grado (Spearman 0.75), pero solo moderadamente con seguidores (0.36) y popularidad (0.37): ser central no es ser el más escuchado.
- **HITS no dirigido:** hub = authority (`AᵀA = AAᵀ`), y se concentra en el núcleo denso de reggaeton/trap y hip hop.
- **Tópicos de nicho** (reggaeton 0.37, k-pop 0.44) se alejan más del ranking general que los amplios (hip hop 0.64, pop 0.72) y revelan artistas *puente* entre escenas.

## ⚛️ Parte II · Red de citas de física de altas energías

Los nodos son artículos de arXiv HEP-PH (1993–2003) y una arista `A → B` significa que A cita a B.
Como esta red sí es dirigida, hubs y authorities se distinguen de forma natural. Los tópicos se
definen por **comunidades (Louvain)** y por **época**.

**Hallazgos principales**

- 34,546 artículos y 421,534 citas; 12.2 citas por artículo en promedio (máximo 846).
- **PageRank ≈ citas** (Spearman 0.91), con **sesgo hacia lo antiguo** (−0.38 con el año): las citas apuntan al pasado.
- **HITS dirigido** encuentra el subtema de moda: las authorities son los artículos fundacionales de **dimensiones extra** (1998–1999, p. ej. `hep-ph/9803315`) y los hubs son **revisiones** con muchas referencias. PageRank y HITS comparten solo 8 artículos en su top-100.
- **Tópico por época (1999–2002):** sube `hep-ph/9803315` al 1.er lugar del ranking.

> 🔁 **Comparación:** en ambas redes PageRank y HITS miden cosas distintas (alcance global vs. núcleo denso),
> y el PageRank por tópico revela referentes y puentes que el ranking general oculta.

## 🌐 Parte III · Wikipedia a gran escala

Un pipeline que convierte el dump XML de Wikipedia en inglés en un grafo de **7,249,098 nodos y
236,203,272 aristas**, y le aplica PageRank, HITS (global y por consulta de Kleinberg) y PageRank
sensible a tópico (Matemáticas, Física, Computación, Música, Deportes, Biología).

```
 XML (118 GB) ──▶ extract_links.py ──▶ Parquet (3.4 GB) ──▶ build_graph.py ──▶ Grafo (2.0 GB) ──▶ Notebook
                  12 procesos,                              Polars,                              SciPy CSR
                  streaming                                 redirecciones                        PageRank · HITS · Tópicos
```

| Paso | Herramienta | Tiempo | Resultado |
|---|---|---|---|
| 1. Extracción | `multiprocessing` + regex | ~2.5 min | 19.3M páginas |
| 2. Grafo | Polars (streaming) | ~25 s, 7.5 GB RAM pico | 7.25M nodos, 236M aristas |
| 3. Análisis | NumPy / SciPy sparse | ~25 min | Rankings y gráficas |

**Decisiones de diseño** (16 GB de RAM frente a 118 GB de datos): lectura del XML en *streaming* por
rangos de bytes, grafo como **matriz dispersa CSR** en lugar de NetworkX, y resultados intermedios en
**Parquet**. Los resultados se **validaron contra NetworkX** en un subgrafo de 3,000 nodos
(PageRank con diferencia máxima de 4·10⁻¹⁰ y HITS con Spearman ρ = 1.0).

Los detalles completos están en [`PIPELINE.md`](PIPELINE.md).

---

## 🚀 Cómo reproducirlo

```bash
# 1. Entorno
uv venv p3 --python 3.12
uv pip install --python p3 -r requirements.txt
p3/bin/python -m ipykernel install --user --name p3 --display-name "Python (p3)"
```

**Partes I y II**: descarga `nodes.csv` y `edges.csv` ([Kaggle](https://www.kaggle.com/)) y `cit-HepPh.txt` y
`cit-HepPh-dates.txt` ([SNAP](https://snap.stanford.edu/data/cit-HepPh.html)) en la raíz, y abre:

```bash
p3/bin/jupyter lab analisis_pagerank_hits.ipynb
```

**Parte III**: descarga el dump `enwiki-latest-pages-articles-multistream.xml` en `data/` y ejecuta:

```bash
p3/bin/python wikipedia/src/extract_links.py data/enwiki-latest-pages-articles-multistream.xml data/parquet --chunks 192
p3/bin/python wikipedia/src/build_graph.py data/parquet data/graph
p3/bin/jupyter lab wikipedia/notebooks/analisis_enlaces_wikipedia.ipynb
```

Para pruebas rápidas, `--limit-bytes 1200000000` procesa solo los primeros ~1.2 GB del XML y
`--workers N` limita el número de procesos.

## 👥 Equipo

- Angel Noel Perez Martinez
- Isaac Monroy Villegas
- Valentina Ayelen Cruz Mendoza

## 📚 Referencias

- Page, L., Brin, S., Motwani, R., Winograd, T. (1999). *The PageRank Citation Ranking*.
- Kleinberg, J. (1999). *Authoritative sources in a hyperlinked environment*.
- Haveliwala, T. (2002). *Topic-Sensitive PageRank*.
