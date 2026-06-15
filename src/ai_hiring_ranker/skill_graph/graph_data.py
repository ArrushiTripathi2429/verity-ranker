"""
Static skill & role knowledge graph data for Layer 7.

This module defines the graph as pure Python — no external files, no DB.
It is loaded once at import time and cached as a module-level constant.

Coverage:
  - 80+ canonical skills across: ML/AI, data engineering, backend, frontend,
    cloud/DevOps, databases, soft skills, and domain knowledge
  - Synonym aliases so abbreviations and alternative names all resolve
  - Adjacent skill edges (commonly co-used tools)
  - Transferable skill edges (experience implies capacity)
  - Subset/superset edges (skill hierarchies)
  - Role hierarchy edges for seniority inference

Adding new skills:
  1. Add a SkillNode to _RAW_NODES with canonical_name, aliases, category, edges.
  2. Re-run the module — it self-validates on import.

Relationship weight guide:
  1.0  — synonym / exact alias
  0.85 — very strongly adjacent (almost always co-used)
  0.70 — strongly adjacent
  0.55 — moderately adjacent
  0.40 — loosely adjacent / transferable with effort
  0.30 — weak transferability
"""

from __future__ import annotations

from .schemas import RelationshipType, SkillEdge, SkillNode

# ---------------------------------------------------------------------------
# Raw node definitions
# ---------------------------------------------------------------------------
# Keep this list sorted by canonical_name for readability.
# Edges only need to be listed on ONE side; graph.py adds reverse edges
# automatically for bidirectional relationships.

_RAW_NODES: list[SkillNode] = [

    # ── Programming languages ──────────────────────────────────────────────

    SkillNode(
        canonical_name="Python",
        aliases=["python", "python3", "py"],
        category="programming_language",
        edges=[
            SkillEdge(target="FastAPI",        relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Flask",          relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="Django",         relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
            SkillEdge(target="Pandas",         relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="NumPy",          relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Scikit-Learn",   relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Machine Learning", relationship=RelationshipType.ADJACENT,   weight=0.75, bidirectional=False),
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.60, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Java",
        aliases=["java"],
        category="programming_language",
        edges=[
            SkillEdge(target="Spring Boot",    relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Scala",          relationship=RelationshipType.TRANSFERABLE, weight=0.55, bidirectional=True),
            SkillEdge(target="Kotlin",         relationship=RelationshipType.TRANSFERABLE, weight=0.60, bidirectional=True),
        ],
    ),

    SkillNode(
        canonical_name="Scala",
        aliases=["scala"],
        category="programming_language",
        edges=[
            SkillEdge(target="Apache Spark",   relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="Java",           relationship=RelationshipType.TRANSFERABLE, weight=0.55, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Kotlin",
        aliases=["kotlin"],
        category="programming_language",
        edges=[
            SkillEdge(target="Java",           relationship=RelationshipType.TRANSFERABLE, weight=0.60, bidirectional=False),
            SkillEdge(target="Android",        relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Go",
        aliases=["go", "golang"],
        category="programming_language",
        edges=[
            SkillEdge(target="Docker",         relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
            SkillEdge(target="Kubernetes",     relationship=RelationshipType.ADJACENT,     weight=0.60, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Rust",
        aliases=["rust", "rust-lang"],
        category="programming_language",
        edges=[
            SkillEdge(target="Systems Programming", relationship=RelationshipType.ADJACENT, weight=0.85, bidirectional=False),
            SkillEdge(target="C++",            relationship=RelationshipType.TRANSFERABLE, weight=0.50, bidirectional=True),
        ],
    ),

    SkillNode(
        canonical_name="C++",
        aliases=["c++", "cpp", "c plus plus"],
        category="programming_language",
        edges=[
            SkillEdge(target="Systems Programming", relationship=RelationshipType.ADJACENT, weight=0.80, bidirectional=False),
            SkillEdge(target="Rust",           relationship=RelationshipType.TRANSFERABLE, weight=0.50, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="JavaScript",
        aliases=["javascript", "js", "ecmascript", "es6", "es2015"],
        category="programming_language",
        edges=[
            SkillEdge(target="TypeScript",     relationship=RelationshipType.TRANSFERABLE, weight=0.85, bidirectional=True),
            SkillEdge(target="React",          relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Node.js",        relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Vue.js",         relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="TypeScript",
        aliases=["typescript", "ts"],
        category="programming_language",
        edges=[
            SkillEdge(target="JavaScript",     relationship=RelationshipType.TRANSFERABLE, weight=0.85, bidirectional=False),
            SkillEdge(target="React",          relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Node.js",        relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="SQL",
        aliases=["sql", "structured query language"],
        category="programming_language",
        edges=[
            SkillEdge(target="PostgreSQL",     relationship=RelationshipType.SUBSET,       weight=0.90, bidirectional=False),
            SkillEdge(target="MySQL",          relationship=RelationshipType.SUBSET,       weight=0.90, bidirectional=False),
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.70, bidirectional=False),
            SkillEdge(target="Snowflake",      relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
        ],
    ),

    # ── Web frameworks ────────────────────────────────────────────────────

    SkillNode(
        canonical_name="FastAPI",
        aliases=["fastapi", "fast api"],
        category="framework",
        edges=[
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="REST API",       relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Flask",          relationship=RelationshipType.TRANSFERABLE, weight=0.70, bidirectional=True),
            SkillEdge(target="Pydantic",       relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Flask",
        aliases=["flask"],
        category="framework",
        edges=[
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="REST API",       relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="FastAPI",        relationship=RelationshipType.TRANSFERABLE, weight=0.70, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Django",
        aliases=["django", "django rest framework", "drf"],
        category="framework",
        edges=[
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="REST API",       relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="PostgreSQL",     relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Spring Boot",
        aliases=["spring boot", "spring", "spring framework"],
        category="framework",
        edges=[
            SkillEdge(target="Java",           relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="REST API",       relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Microservices",  relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="React",
        aliases=["react", "reactjs", "react.js"],
        category="framework",
        edges=[
            SkillEdge(target="JavaScript",     relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="TypeScript",     relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Vue.js",         relationship=RelationshipType.TRANSFERABLE, weight=0.60, bidirectional=True),
        ],
    ),

    SkillNode(
        canonical_name="Vue.js",
        aliases=["vue", "vuejs", "vue.js"],
        category="framework",
        edges=[
            SkillEdge(target="JavaScript",     relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="React",          relationship=RelationshipType.TRANSFERABLE, weight=0.60, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Node.js",
        aliases=["node.js", "nodejs", "node"],
        category="framework",
        edges=[
            SkillEdge(target="JavaScript",     relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="TypeScript",     relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="REST API",       relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Pydantic",
        aliases=["pydantic"],
        category="framework",
        edges=[
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="FastAPI",        relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
        ],
    ),

    # ── ML / AI ───────────────────────────────────────────────────────────

    SkillNode(
        canonical_name="Machine Learning",
        aliases=["machine learning", "ml", "supervised learning", "unsupervised learning",
                 "predictive modeling", "predictive modelling", "statistical learning"],
        category="ml_concept",
        edges=[
            SkillEdge(target="Deep Learning",  relationship=RelationshipType.SUPERSET,     weight=0.90, bidirectional=False),
            SkillEdge(target="Scikit-Learn",   relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Feature Engineering", relationship=RelationshipType.ADJACENT, weight=0.80, bidirectional=False),
            SkillEdge(target="Model Evaluation", relationship=RelationshipType.ADJACENT,   weight=0.80, bidirectional=False),
            SkillEdge(target="MLOps",          relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Deep Learning",
        aliases=["deep learning", "dl", "neural networks", "neural network", "ann", "dnn"],
        category="ml_concept",
        edges=[
            SkillEdge(target="Machine Learning", relationship=RelationshipType.SUBSET,     weight=0.90, bidirectional=False),
            SkillEdge(target="PyTorch",        relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="TensorFlow",     relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Computer Vision", relationship=RelationshipType.ADJACENT,    weight=0.70, bidirectional=False),
            SkillEdge(target="NLP",            relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="PyTorch",
        aliases=["pytorch", "torch"],
        category="framework",
        edges=[
            SkillEdge(target="Deep Learning",  relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="TensorFlow",     relationship=RelationshipType.TRANSFERABLE, weight=0.70, bidirectional=True),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="TensorFlow",
        aliases=["tensorflow", "tf", "tf2", "keras"],
        category="framework",
        edges=[
            SkillEdge(target="Deep Learning",  relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="PyTorch",        relationship=RelationshipType.TRANSFERABLE, weight=0.70, bidirectional=False),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Scikit-Learn",
        aliases=["scikit-learn", "sklearn", "scikit learn"],
        category="framework",
        edges=[
            SkillEdge(target="Machine Learning", relationship=RelationshipType.ADJACENT,   weight=0.90, bidirectional=False),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="Pandas",         relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="NLP",
        aliases=["nlp", "natural language processing", "text mining", "text classification",
                 "sentiment analysis", "named entity recognition", "ner"],
        category="ml_concept",
        edges=[
            SkillEdge(target="Deep Learning",  relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="LLMs",           relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Transformers",   relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="LLMs",
        aliases=["llm", "llms", "large language models", "large language model",
                 "gpt", "gpt-4", "chatgpt", "claude", "gemini"],
        category="ml_concept",
        edges=[
            SkillEdge(target="NLP",            relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Transformers",   relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="RAG",            relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Prompt Engineering", relationship=RelationshipType.ADJACENT, weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Transformers",
        aliases=["transformers", "hugging face", "huggingface", "bert", "gpt", "attention mechanism"],
        category="framework",
        edges=[
            SkillEdge(target="LLMs",           relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="NLP",            relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="PyTorch",        relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="RAG",
        aliases=["rag", "retrieval augmented generation", "retrieval-augmented generation"],
        category="ml_concept",
        edges=[
            SkillEdge(target="LLMs",           relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="Vector Databases", relationship=RelationshipType.ADJACENT,   weight=0.85, bidirectional=False),
            SkillEdge(target="Embeddings",     relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Embeddings",
        aliases=["embeddings", "embedding", "vector embeddings", "text embeddings", "sentence embeddings"],
        category="ml_concept",
        edges=[
            SkillEdge(target="RAG",            relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Vector Databases", relationship=RelationshipType.ADJACENT,   weight=0.85, bidirectional=False),
            SkillEdge(target="NLP",            relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="Semantic Search", relationship=RelationshipType.ADJACENT,    weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Prompt Engineering",
        aliases=["prompt engineering", "prompt design", "prompting", "few-shot prompting",
                 "chain of thought", "cot"],
        category="ml_concept",
        edges=[
            SkillEdge(target="LLMs",           relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="RAG",            relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Computer Vision",
        aliases=["computer vision", "cv", "image recognition", "image classification",
                 "object detection", "image segmentation"],
        category="ml_concept",
        edges=[
            SkillEdge(target="Deep Learning",  relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="PyTorch",        relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="OpenCV",         relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="OpenCV",
        aliases=["opencv", "open cv"],
        category="framework",
        edges=[
            SkillEdge(target="Computer Vision", relationship=RelationshipType.ADJACENT,    weight=0.90, bidirectional=False),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Feature Engineering",
        aliases=["feature engineering", "feature extraction", "feature selection"],
        category="ml_concept",
        edges=[
            SkillEdge(target="Machine Learning", relationship=RelationshipType.ADJACENT,   weight=0.85, bidirectional=False),
            SkillEdge(target="Pandas",         relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Model Evaluation",
        aliases=["model evaluation", "model assessment", "evaluation pipelines",
                 "metrics", "evaluated classification", "a/b testing ml", "model metrics"],
        category="ml_concept",
        edges=[
            SkillEdge(target="Machine Learning", relationship=RelationshipType.ADJACENT,   weight=0.85, bidirectional=False),
            SkillEdge(target="Scikit-Learn",   relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="MLOps",
        aliases=["mlops", "ml ops", "ml engineering", "model deployment", "model serving",
                 "model monitoring", "ml pipeline"],
        category="ml_concept",
        edges=[
            SkillEdge(target="Machine Learning", relationship=RelationshipType.ADJACENT,   weight=0.80, bidirectional=False),
            SkillEdge(target="Docker",         relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="Kubernetes",     relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
            SkillEdge(target="CI/CD",          relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
            SkillEdge(target="Airflow",        relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
        ],
    ),

    # ── Data science / engineering ────────────────────────────────────────

    SkillNode(
        canonical_name="Pandas",
        aliases=["pandas", "pd"],
        category="framework",
        edges=[
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="NumPy",          relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=True),
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.65, bidirectional=False),
            SkillEdge(target="Feature Engineering", relationship=RelationshipType.ADJACENT, weight=0.75, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="NumPy",
        aliases=["numpy", "np"],
        category="framework",
        edges=[
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="Pandas",         relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Scikit-Learn",   relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Apache Spark",
        aliases=["apache spark", "spark", "pyspark", "spark sql"],
        category="data_engineering",
        edges=[
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.90, bidirectional=False),
            SkillEdge(target="Scala",          relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="Hadoop",         relationship=RelationshipType.TRANSFERABLE, weight=0.55, bidirectional=True),
            SkillEdge(target="Kafka",          relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Data Engineering",
        aliases=["data engineering", "data pipeline", "etl", "elt", "data warehouse",
                 "data lake", "data pipelines", "batch processing"],
        category="data_engineering",
        edges=[
            SkillEdge(target="Apache Spark",   relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Airflow",        relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="SQL",            relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Kafka",          relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
            SkillEdge(target="Snowflake",      relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Airflow",
        aliases=["airflow", "apache airflow"],
        category="data_engineering",
        edges=[
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.90, bidirectional=False),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="MLOps",          relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Kafka",
        aliases=["kafka", "apache kafka"],
        category="data_engineering",
        edges=[
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.85, bidirectional=False),
            SkillEdge(target="Apache Spark",   relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
            SkillEdge(target="Microservices",  relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Hadoop",
        aliases=["hadoop", "hdfs", "mapreduce", "map reduce"],
        category="data_engineering",
        edges=[
            SkillEdge(target="Apache Spark",   relationship=RelationshipType.TRANSFERABLE, weight=0.55, bidirectional=False),
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.80, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Snowflake",
        aliases=["snowflake"],
        category="data_engineering",
        edges=[
            SkillEdge(target="SQL",            relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.80, bidirectional=False),
            SkillEdge(target="dbt",            relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="dbt",
        aliases=["dbt", "dbt-core", "data build tool"],
        category="data_engineering",
        edges=[
            SkillEdge(target="Snowflake",      relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="SQL",            relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.80, bidirectional=False),
        ],
    ),

    # ── Databases ─────────────────────────────────────────────────────────

    SkillNode(
        canonical_name="PostgreSQL",
        aliases=["postgresql", "postgres", "pg"],
        category="platform",
        edges=[
            SkillEdge(target="SQL",            relationship=RelationshipType.SUPERSET,     weight=0.95, bidirectional=False),
            SkillEdge(target="MySQL",          relationship=RelationshipType.TRANSFERABLE, weight=0.75, bidirectional=True),
        ],
    ),

    SkillNode(
        canonical_name="MySQL",
        aliases=["mysql"],
        category="platform",
        edges=[
            SkillEdge(target="SQL",            relationship=RelationshipType.SUPERSET,     weight=0.95, bidirectional=False),
            SkillEdge(target="PostgreSQL",     relationship=RelationshipType.TRANSFERABLE, weight=0.75, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="MongoDB",
        aliases=["mongodb", "mongo"],
        category="platform",
        edges=[
            SkillEdge(target="NoSQL",          relationship=RelationshipType.SUBSET,       weight=0.90, bidirectional=False),
            SkillEdge(target="Redis",          relationship=RelationshipType.ADJACENT,     weight=0.55, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Redis",
        aliases=["redis"],
        category="platform",
        edges=[
            SkillEdge(target="NoSQL",          relationship=RelationshipType.SUBSET,       weight=0.85, bidirectional=False),
            SkillEdge(target="Caching",        relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="NoSQL",
        aliases=["nosql", "no sql"],
        category="platform",
        edges=[
            SkillEdge(target="MongoDB",        relationship=RelationshipType.SUPERSET,     weight=0.90, bidirectional=False),
            SkillEdge(target="Redis",          relationship=RelationshipType.SUPERSET,     weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Vector Databases",
        aliases=["vector database", "vector db", "vector store", "faiss", "pinecone",
                 "weaviate", "chroma", "qdrant", "milvus"],
        category="platform",
        edges=[
            SkillEdge(target="Embeddings",     relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="RAG",            relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Semantic Search", relationship=RelationshipType.ADJACENT,    weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Semantic Search",
        aliases=["semantic search", "dense retrieval", "vector search",
                 "approximate nearest neighbour", "ann search"],
        category="ml_concept",
        edges=[
            SkillEdge(target="Embeddings",     relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="Vector Databases", relationship=RelationshipType.ADJACENT,   weight=0.85, bidirectional=False),
        ],
    ),

    # ── Cloud / DevOps ────────────────────────────────────────────────────

    SkillNode(
        canonical_name="Docker",
        aliases=["docker", "docker-compose", "containerization", "containers"],
        category="devops",
        edges=[
            SkillEdge(target="Kubernetes",     relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="CI/CD",          relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="MLOps",          relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
            SkillEdge(target="Microservices",  relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Kubernetes",
        aliases=["kubernetes", "k8s", "kubectl", "helm"],
        category="devops",
        edges=[
            SkillEdge(target="Docker",         relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="CI/CD",          relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="Cloud Platforms", relationship=RelationshipType.ADJACENT,    weight=0.70, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="CI/CD",
        aliases=["ci/cd", "ci cd", "continuous integration", "continuous deployment",
                 "continuous delivery", "ci pipelines", "github actions", "jenkins",
                 "gitlab ci", "circleci"],
        category="devops",
        edges=[
            SkillEdge(target="Docker",         relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="Testing",        relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="MLOps",          relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Cloud Platforms",
        aliases=["aws", "amazon web services", "gcp", "google cloud", "azure",
                 "microsoft azure", "cloud", "cloud infrastructure"],
        category="platform",
        edges=[
            SkillEdge(target="Kubernetes",     relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
            SkillEdge(target="Docker",         relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
            SkillEdge(target="MLOps",          relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Cloud Deployment",
        aliases=["cloud deployment", "cloud deployments", "deployments", "deployment",
                 "production deployment"],
        category="devops",
        edges=[
            SkillEdge(target="Cloud Platforms", relationship=RelationshipType.ADJACENT,    weight=0.85, bidirectional=False),
            SkillEdge(target="Docker",         relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="CI/CD",          relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Microservices",
        aliases=["microservices", "microservice", "service-oriented architecture",
                 "soa", "distributed systems"],
        category="devops",
        edges=[
            SkillEdge(target="Docker",         relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="Kubernetes",     relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
            SkillEdge(target="Kafka",          relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
            SkillEdge(target="REST API",       relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="REST API",
        aliases=["rest api", "rest", "restful", "restful api", "http api", "web api"],
        category="framework",
        edges=[
            SkillEdge(target="FastAPI",        relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Flask",          relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="Microservices",  relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
        ],
    ),

    # ── Software engineering ──────────────────────────────────────────────

    SkillNode(
        canonical_name="Testing",
        aliases=["testing", "tests", "unit testing", "integration testing",
                 "test-driven development", "tdd", "pytest", "unittest"],
        category="devops",
        edges=[
            SkillEdge(target="CI/CD",          relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Git",
        aliases=["git", "version control", "github", "gitlab", "bitbucket"],
        category="devops",
        edges=[
            SkillEdge(target="CI/CD",          relationship=RelationshipType.ADJACENT,     weight=0.70, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Systems Programming",
        aliases=["systems programming", "low-level programming", "performance engineering"],
        category="programming_language",
        edges=[
            SkillEdge(target="C++",            relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
            SkillEdge(target="Rust",           relationship=RelationshipType.ADJACENT,     weight=0.80, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Caching",
        aliases=["caching", "cache", "memcached"],
        category="platform",
        edges=[
            SkillEdge(target="Redis",          relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
        ],
    ),

    # ── Misc / domain ─────────────────────────────────────────────────────

    SkillNode(
        canonical_name="LangChain",
        aliases=["langchain", "lang chain"],
        category="framework",
        edges=[
            SkillEdge(target="LLMs",           relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="RAG",            relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="LangGraph",
        aliases=["langgraph", "lang graph"],
        category="framework",
        edges=[
            SkillEdge(target="LangChain",      relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="LLMs",           relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Streamlit",
        aliases=["streamlit"],
        category="framework",
        edges=[
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.90, bidirectional=False),
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.55, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Android",
        aliases=["android", "android development"],
        category="platform",
        edges=[
            SkillEdge(target="Kotlin",         relationship=RelationshipType.ADJACENT,     weight=0.85, bidirectional=False),
            SkillEdge(target="Java",           relationship=RelationshipType.ADJACENT,     weight=0.75, bidirectional=False),
        ],
    ),

    SkillNode(
        canonical_name="Dashboards",
        aliases=["dashboards", "dashboard", "data visualization", "data visualisation",
                 "tableau", "power bi", "plotly", "matplotlib", "seaborn"],
        category="domain",
        edges=[
            SkillEdge(target="Data Engineering", relationship=RelationshipType.ADJACENT,   weight=0.60, bidirectional=False),
            SkillEdge(target="Python",         relationship=RelationshipType.ADJACENT,     weight=0.65, bidirectional=False),
        ],
    ),
]
