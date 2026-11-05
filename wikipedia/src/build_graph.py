"""Construye el grafo dirigido de artículos a partir de los Parquet extraídos.

Salidas en OUT_DIR:
  nodes.parquet  : idx, page_id, title          (solo artículos, sin redirecciones)
  edges.parquet  : src, dst (int32, únicos, sin lazos), redirecciones resueltas
  cats.parquet   : idx, category

Uso: python src/build_graph.py data/parquet data/graph
"""
import os
import sys
import time

import polars as pl

src_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

pages = pl.read_parquet(f"{src_dir}/pages_*.parquet").unique("page_id")
nodes = (pages.filter(pl.col("redirect").is_null())
         .unique("title", keep="first").sort("page_id")
         .with_row_index("idx").select(pl.col("idx").cast(pl.Int32), "page_id", "title"))
nodes.write_parquet(f"{out_dir}/nodes.parquet")
log(f"{nodes.height:,} artículos")

# título -> idx, resolviendo redirecciones (hasta 2 saltos)
title2idx = nodes.select("title", "idx")
redir = pages.filter(pl.col("redirect").is_not_null()).select("title", "redirect")
r1 = redir.join(title2idx, left_on="redirect", right_on="title", how="inner")
r2 = (redir.join(redir.rename({"title": "mid", "redirect": "r2"}),
                 left_on="redirect", right_on="mid")
      .join(title2idx, left_on="r2", right_on="title"))
resolve = (pl.concat([title2idx,
                      r1.select("title", "idx"),
                      r2.select("title", "idx")])
           .unique("title", keep="first"))
log(f"{resolve.height:,} títulos resolubles ({redir.height:,} redirecciones)")
pid2idx = nodes.select("page_id", "idx")

edges = (pl.scan_parquet(f"{src_dir}/links_*.parquet")
         .rename({"dst": "dst_title"})
         .join(pid2idx.lazy(), left_on="src_id", right_on="page_id")
         .join(resolve.lazy().rename({"idx": "dst"}), left_on="dst_title", right_on="title")
         .select(pl.col("idx").alias("src"), "dst")
         .filter(pl.col("src") != pl.col("dst"))
         .unique())
edges.sink_parquet(f"{out_dir}/edges.parquet")
n_e = pl.scan_parquet(f"{out_dir}/edges.parquet").select(pl.len()).collect().item()
log(f"{n_e:,} aristas")

(pl.scan_parquet(f"{src_dir}/cats_*.parquet")
 .join(pid2idx.lazy(), on="page_id").select("idx", "category").unique()
 .sink_parquet(f"{out_dir}/cats.parquet"))
log("categorías listas")
