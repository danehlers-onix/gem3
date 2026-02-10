#!/usr/bin/env python3
"""
gem3 — Visual RAG: Infinite Zoom Knowledge Navigation

Usage:
  python main.py rag                        Launch grounded zoom server (pre-built tree)
  python main.py search "your query"        Visual agent search with explainability MP4
  python main.py zoom [query]               Launch infinite generative zoom
  python main.py export [query]             Export standalone HTML (host anywhere, zero cost)
  python main.py ingest <data_dir>          Ingest documents into knowledge hierarchy
  python main.py demo                       Run the built-in demo with sample data

Deployment:
  ./deploy.sh fly                           Deploy to Fly.io (free tier)
  ./deploy.sh cloudrun                      Deploy to Google Cloud Run
  ./deploy.sh docker                        Build & run locally in Docker
"""

import sys
import json
import pickle
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.markdown import Markdown

from gem3.config import DATA_DIR, OUTPUT_DIR
from gem3.models import KnowledgeNode

console = Console()
KNOWLEDGE_STORE = OUTPUT_DIR / "knowledge_tree.pkl"


def print_tree(node: KnowledgeNode, tree: Tree | None = None, max_depth: int = 4, depth: int = 0):
    """Pretty-print the knowledge hierarchy."""
    if depth >= max_depth:
        return
    
    label = f"[bold cyan]{node.level}[/] {node.title}"
    if node.summary:
        label += f" [dim]— {node.summary[:60]}[/]"
    label += f" [dim italic]({node.id})[/]"
    
    if tree is None:
        tree = Tree(label)
    else:
        tree = tree.add(label)
    
    for child in node.children:
        print_tree(child, tree, max_depth, depth + 1)
    
    if depth == 0:
        console.print(tree)


def cmd_ingest(data_dir: str):
    """Ingest documents from a directory."""
    from gem3.ingest import ingest_directory
    
    path = Path(data_dir)
    if not path.exists():
        console.print(f"[red]Error:[/] Directory '{data_dir}' not found.")
        sys.exit(1)
    
    console.print(Panel.fit(
        f"[bold]Ingesting documents from:[/] {path.resolve()}",
        title="📚 gem3 Ingest",
    ))
    
    with console.status("[bold green]Building knowledge hierarchy with Gemini..."):
        root = ingest_directory(path)
    
    # Save the knowledge tree
    with open(KNOWLEDGE_STORE, "wb") as f:
        pickle.dump(root, f)
    
    console.print(f"\n[green]✓[/] Knowledge tree built and saved to {KNOWLEDGE_STORE}")
    console.print()
    print_tree(root)
    
    # Count nodes
    def count_nodes(n):
        return 1 + sum(count_nodes(c) for c in n.children)
    
    console.print(f"\n[dim]Total nodes: {count_nodes(root)}[/]")


def cmd_query(query: str):
    """Run a visual RAG query."""
    from gem3.navigator import navigate_visual
    
    # Load the knowledge tree
    if not KNOWLEDGE_STORE.exists():
        console.print("[red]Error:[/] No knowledge tree found. Run 'ingest' first.")
        sys.exit(1)
    
    with open(KNOWLEDGE_STORE, "rb") as f:
        root = pickle.load(f)
    
    console.print(Panel.fit(
        f"[bold]Query:[/] {query}",
        title=" gem3 Visual Navigation",
    ))
    
    result = navigate_visual(query, root, verbose=True)
    
    # Print results
    console.print(Panel.fit(
        Markdown(result.synthesis) if result.synthesis else "[dim]No synthesis generated[/]",
        title=" Answer",
        border_style="green",
    ))
    
    console.print(f"\n[dim]Zoom trace: {len(result.zoom_trace)} steps[/]")
    console.print(f"[dim]Retrieved: {len(result.retrieved_nodes)} nodes[/]")
    console.print(f"[dim]Screenshots saved to: {OUTPUT_DIR / 'screenshots'}[/]")
    console.print(f"[dim]HTML views saved to: {OUTPUT_DIR / 'html'}[/]")


def cmd_demo():
    """Run with built-in sample data."""
    from gem3.ingest import ingest_documents
    from gem3.navigator import navigate_visual
    
    console.print(Panel.fit(
        "[bold]gem3 Visual RAG Demo[/]\n"
        "Infinite Zoom Knowledge Navigation\n"
        "[dim]No embeddings. No keywords. Pure vision.[/]",
        title=" gem3",
        border_style="blue",
    ))
    
    # Sample documents about various science topics
    documents = [
        {
            "title": "Photosynthesis: Light to Life",
            "content": """Photosynthesis is the process by which green plants, algae, and certain bacteria 
convert light energy into chemical energy stored in glucose. This process occurs primarily in 
the chloroplasts of plant cells, specifically within structures called thylakoids.

The process consists of two main stages:

Light-Dependent Reactions: These occur in the thylakoid membranes. When chlorophyll absorbs 
sunlight, it energizes electrons that pass through an electron transport chain. This process 
splits water molecules (photolysis), releasing oxygen as a byproduct. The energy is used to 
produce ATP and NADPH.

The Calvin Cycle (Light-Independent Reactions): Taking place in the stroma of the chloroplast, 
this cycle uses the ATP and NADPH from the light reactions to fix carbon dioxide into organic 
molecules through a process called carbon fixation. The enzyme RuBisCO catalyzes the first 
step, combining CO2 with a 5-carbon sugar (RuBP).

The overall equation: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2

Environmental factors affecting photosynthesis include light intensity, CO2 concentration, 
temperature, and water availability. C4 and CAM plants have evolved alternative pathways 
to minimize photorespiration in hot, dry environments.""",
        },
        {
            "title": "Neural Networks: From Biology to Silicon",
            "content": """Artificial neural networks are computational systems inspired by biological neural 
networks in the brain. They consist of interconnected nodes (neurons) organized in layers 
that process information using connectionist approaches.

Architecture Basics:
- Input Layer: Receives raw data (pixels, text tokens, sensor readings)
- Hidden Layers: Transform data through weighted connections and activation functions
- Output Layer: Produces predictions or classifications

Key Concepts:
Backpropagation is the algorithm used to train neural networks. It calculates the gradient 
of the loss function with respect to each weight, then adjusts weights to minimize error. 
This process uses the chain rule of calculus.

Activation Functions determine whether a neuron should fire:
- ReLU: f(x) = max(0, x) — most commonly used, avoids vanishing gradients
- Sigmoid: f(x) = 1/(1+e^-x) — outputs between 0 and 1
- Tanh: f(x) = (e^x - e^-x)/(e^x + e^-x) — outputs between -1 and 1

Modern architectures include:
- CNNs (Convolutional Neural Networks) for image processing
- RNNs/LSTMs for sequential data
- Transformers for attention-based processing (GPT, BERT)
- GANs for generative tasks

The transformer architecture, introduced in "Attention Is All You Need" (2017), 
revolutionized NLP through self-attention mechanisms that process all tokens simultaneously.""",
        },
        {
            "title": "Quantum Entanglement: Spooky Action at a Distance",
            "content": """Quantum entanglement is a phenomenon where two or more particles become correlated 
in such a way that the quantum state of each particle cannot be described independently. 
When particles are entangled, measuring one particle instantaneously affects the other, 
regardless of distance — what Einstein called "spooky action at a distance."

How Entanglement Works:
When two particles interact and become entangled, their quantum states become linked. 
For example, if two electrons are entangled in a singlet state, measuring one as spin-up 
means the other will always be spin-down, even if they are light-years apart.

Bell's Theorem (1964): John Stewart Bell proved that no theory of local hidden variables 
can reproduce all predictions of quantum mechanics. Experiments by Alain Aspect (1982) 
and others have confirmed quantum mechanics' predictions, ruling out local hidden variables.

Applications:
- Quantum Computing: Entangled qubits enable quantum gates and quantum parallelism
- Quantum Cryptography: QKD (Quantum Key Distribution) uses entanglement for 
  theoretically unbreakable encryption (BB84 protocol)
- Quantum Teleportation: Transferring quantum states (not matter) using entanglement 
  and classical communication
- Quantum Sensing: Enhanced measurement precision beyond classical limits

The EPR Paradox: Einstein, Podolsky, and Rosen argued in 1935 that quantum mechanics 
must be incomplete because entanglement seemed to violate locality. However, entanglement 
does not enable faster-than-light communication because measurement results appear random 
without classical information exchange.""",
        },
        {
            "title": "CRISPR-Cas9: Editing the Code of Life",
            "content": """CRISPR-Cas9 is a revolutionary gene-editing technology adapted from a natural 
defense system found in bacteria. It allows scientists to precisely modify DNA sequences 
in living organisms with unprecedented ease and accuracy.

How CRISPR Works:
1. Guide RNA (gRNA): A short RNA sequence (~20 nucleotides) is designed to match the 
   target DNA sequence. This guide RNA directs the Cas9 protein to the right location.
2. Cas9 Protein: This molecular scissors creates a double-strand break in the DNA at 
   the target site specified by the guide RNA.
3. DNA Repair: The cell's natural repair mechanisms kick in:
   - NHEJ (Non-Homologous End Joining): Error-prone, often disrupts the gene
   - HDR (Homology-Directed Repair): Uses a template to make precise edits

Applications in Medicine:
- Sickle Cell Disease: Clinical trials using CRISPR to edit the BCL11A gene, enabling 
  fetal hemoglobin production. CASGEVY became the first approved CRISPR therapy in 2023.
- Cancer Immunotherapy: Engineering T-cells with CRISPR to better recognize tumors
- Genetic Disorders: Targeting mutations in diseases like cystic fibrosis, muscular dystrophy
- Infectious Disease: Developing CRISPR-based diagnostics (SHERLOCK, DETECTR)

Ethical Considerations:
The case of He Jiankui, who created CRISPR-edited babies in 2018, raised profound ethical 
questions about germline editing — changes that would be inherited by future generations.""",
        },
        {
            "title": "Black Holes: Where Spacetime Breaks Down",
            "content": """A black hole is a region of spacetime where gravity is so intense that nothing — 
not even light — can escape once past the event horizon. They form when massive stars 
collapse at the end of their lives, or through other gravitational processes.

Types of Black Holes:
- Stellar: 5-100 solar masses, formed from collapsed massive stars
- Supermassive: Millions to billions of solar masses, found at galaxy centers
  (Sagittarius A* at the Milky Way center is ~4 million solar masses)
- Intermediate: 100-100,000 solar masses, evidence is growing
- Primordial: Hypothetical, possibly formed in the early universe

Key Physics:
Schwarzschild Radius: r_s = 2GM/c² — the radius of the event horizon for a non-rotating 
black hole. For the Sun, this would be about 3 km.

Hawking Radiation: Stephen Hawking predicted in 1974 that black holes slowly emit 
radiation due to quantum effects near the event horizon. Virtual particle pairs form; 
one falls in, the other escapes as real radiation. This means black holes slowly evaporate.

The Information Paradox: If black holes destroy information (violating quantum mechanics' 
unitarity), or if information is somehow preserved and released, remains one of the 
biggest unsolved problems in theoretical physics.

The Event Horizon Telescope captured the first image of a black hole (M87*) in 2019, 
confirming theoretical predictions about black hole shadows and photon rings.""",
        },
    ]
    
    console.print("\n[bold]Step 1:[/] Ingesting sample documents...\n")
    
    with console.status("[bold green]Building knowledge hierarchy with Gemini 3 Flash..."):
        root = ingest_documents(documents)
    
    # Save it
    with open(KNOWLEDGE_STORE, "wb") as f:
        pickle.dump(root, f)
    
    console.print("[green]✓[/] Knowledge hierarchy built:\n")
    print_tree(root)
    
    # Now run a query
    query = "How do quantum effects relate to information processing?"
    
    console.print(f"\n[bold]Step 2:[/] Visual navigation query: [italic]\"{query}\"[/]\n")
    
    result = navigate_visual(query, root, verbose=True)
    
    console.print(Panel.fit(
        Markdown(result.synthesis) if result.synthesis else "[dim]No synthesis[/]",
        title=" Answer",
        border_style="green",
    ))
    
    console.print(f"\n[bold green]Demo complete![/]")
    console.print(f"[dim]Check {OUTPUT_DIR / 'html'} for generated visual pages[/]")
    console.print(f"[dim]Check {OUTPUT_DIR / 'screenshots'} for navigation screenshots[/]")


def cmd_interactive():
    """Interactive query loop."""
    from gem3.navigator import navigate_visual
    
    if not KNOWLEDGE_STORE.exists():
        console.print("[red]Error:[/] No knowledge tree found. Run 'demo' or 'ingest' first.")
        sys.exit(1)
    
    with open(KNOWLEDGE_STORE, "rb") as f:
        root = pickle.load(f)
    
    console.print(Panel.fit(
        "[bold]gem3 Interactive Mode[/]\n"
        "Type your questions to navigate the knowledge visually.\n"
        "Type 'quit' to exit.",
        title=" gem3",
        border_style="blue",
    ))
    
    while True:
        try:
            query = console.input("\n[bold blue]Query>[/] ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            
            result = navigate_visual(query, root, verbose=True)
            
            console.print(Panel.fit(
                Markdown(result.synthesis) if result.synthesis else "[dim]No synthesis[/]",
                title=" Answer",
                border_style="green",
            ))
        except KeyboardInterrupt:
            break
    
    console.print("\n[dim]Goodbye![/]")


def cmd_zoom(args: list[str]):
    """Launch the infinite generative zoom server."""
    import asyncio
    from server import start_server
    
    topics = None
    query = ""
    
    # Parse args
    i = 0
    while i < len(args):
        if args[i] == "--topics" and i + 1 < len(args):
            topics = [t.strip() for t in args[i + 1].split(",")]
            i += 2
        else:
            query = " ".join(args[i:])
            break
    
    console.print(Panel.fit(
        "[bold]gem3 Infinite Generative Zoom[/]\n"
        "Scroll to zoom into any word.\n"
        "New knowledge is generated by Gemini as you explore.\n"
        "[dim]Every zoom reveals deeper layers — infinitely.[/]",
        title=" gem3",
        border_style="blue",
    ))
    
    asyncio.run(start_server(topics=topics, query=query))


def cmd_search(args: list[str]):
    """
    Visual search: Gemini 3 Flash agent visually navigates the knowledge canvas.
    
    The agent SEES the rendered canvas, decides where to zoom,
    navigates to source excerpts, and builds an explainability graph.
    """
    import pickle
    from gem3.visual_agent import visual_search, print_explainability
    from gem3.ingest import count_nodes, count_grounded
    
    if not args:
        console.print("[red]Usage:[/] python main.py search \"your query\"")
        return
    
    query = " ".join(args)
    
    console.print(Panel.fit(
        f"[bold]gem3 Visual Search Agent[/]\n"
        f"Gemini sees the canvas → decides where to zoom → finds source excerpts\n"
        f"[dim]Query: {query}[/]",
        title=" gem3",
        border_style="cyan",
    ))
    
    # Load knowledge tree
    if not KNOWLEDGE_STORE.exists():
        console.print("\n  No knowledge tree found. Building from demo docs first...")
        from gem3.ingest import ingest_documents
        
        documents = [
            {"title": "Photosynthesis: Light to Life", "content": "Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose. This occurs in chloroplasts within thylakoid structures.\n\nLight-Dependent Reactions occur in thylakoid membranes. Chlorophyll absorbs sunlight, energizing electrons through an electron transport chain. This splits water molecules, releasing oxygen. Energy produces ATP and NADPH.\n\nThe Calvin Cycle uses ATP and NADPH to fix carbon dioxide into organic molecules. The enzyme RuBisCO catalyzes the first step, combining CO2 with RuBP.\n\nThe overall equation: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2\n\nEnvironmental factors include light intensity, CO2 concentration, temperature, and water availability. C4 and CAM plants minimize photorespiration in hot dry environments."},
            {"title": "Quantum Entanglement", "content": "Quantum entanglement is a phenomenon where particles become correlated so the quantum state of each cannot be described independently. Measuring one instantly affects the other regardless of distance.\n\nWhen two particles interact and become entangled, their quantum states become linked. Two entangled electrons in a singlet state: measuring one as spin-up means the other is always spin-down.\n\nBell's Theorem (1964): John Bell proved no theory of local hidden variables can reproduce quantum mechanics predictions. Aspect's experiments (1982) confirmed this.\n\nApplications include quantum computing with entangled qubits, quantum cryptography using QKD for unbreakable encryption, and quantum teleportation transferring quantum states.\n\nThe EPR Paradox: Einstein, Podolsky, Rosen argued quantum mechanics must be incomplete. However, entanglement doesn't enable faster-than-light communication."},
            {"title": "CRISPR-Cas9 Gene Editing", "content": "CRISPR-Cas9 is a gene-editing technology adapted from bacterial defense systems. It allows precise modification of DNA sequences in living organisms.\n\nGuide RNA (gRNA) is a short RNA sequence designed to match the target DNA. It directs the Cas9 protein to the right location in the genome.\n\nCas9 Protein acts as molecular scissors, creating a double-strand break in DNA at the target site. The cell's repair mechanisms then kick in.\n\nDNA Repair uses either NHEJ (error-prone, disrupts genes) or HDR (uses template for precise edits).\n\nMedical applications include sickle cell disease treatment via BCL11A gene editing, cancer immunotherapy with engineered T-cells, and CRISPR-based diagnostics like SHERLOCK."},
            {"title": "Neural Networks and Deep Learning", "content": "Neural networks are computational systems inspired by biological neural networks. They consist of interconnected nodes organized in layers.\n\nArchitecture includes input layers receiving raw data, hidden layers transforming data through weighted connections, and output layers producing predictions.\n\nBackpropagation calculates the gradient of loss with respect to each weight, adjusting weights to minimize error using the chain rule of calculus.\n\nActivation functions include ReLU (max(0,x)), Sigmoid (1/(1+e^-x)), and Tanh. These determine whether neurons should fire.\n\nModern architectures include CNNs for image processing, RNNs for sequential data, and Transformers using self-attention mechanisms for NLP tasks like GPT and BERT."},
            {"title": "Black Holes and Spacetime", "content": "A black hole is a region where gravity is so intense nothing can escape past the event horizon. They form when massive stars collapse.\n\nTypes include stellar (5-100 solar masses), supermassive (millions to billions, at galaxy centers like Sagittarius A*), intermediate, and primordial.\n\nSchwarzschild Radius r_s = 2GM/c² defines the event horizon for non-rotating black holes. For the Sun this would be about 3 km.\n\nHawking Radiation: Black holes slowly emit radiation due to quantum effects near the event horizon. Virtual particle pairs form; one escapes as real radiation.\n\nThe Information Paradox asks whether black holes destroy information, violating quantum unitarity. The Event Horizon Telescope captured the first black hole image (M87*) in 2019."},
        ]
        
        with console.status("[bold green]Building knowledge hierarchy..."):
            root = ingest_documents(documents)
        with open(KNOWLEDGE_STORE, "wb") as f:
            pickle.dump(root, f)
    else:
        with open(KNOWLEDGE_STORE, "rb") as f:
            root = pickle.load(f)
        
        if "x" not in root.metadata:
            from gem3.ingest import _precompute_positions, CANVAS_WIDTH, CANVAS_HEIGHT
            _precompute_positions(root, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)
    
    counts = count_nodes(root)
    n_grounded = count_grounded(root)
    console.print(f"\n  📚 Knowledge tree: {sum(counts.values())} nodes, {n_grounded} grounded excerpts")
    console.print(f"   Starting visual search...\n")
    
    # Run visual agent
    result = visual_search(query, root, max_steps=5, verbose=True)
    
    # Print explainability tree
    console.print()
    print_explainability(result)
    
    # Generate outputs
    from gem3.animate import generate_animation, generate_explainability_html
    from server import find_relevant_paths
    
    highlighted = find_relevant_paths(root, query)
    
    # Generate cinematic MP4/GIF animation
    video_path = generate_animation(result, root, highlighted_ids=highlighted)
    
    # Generate interactive HTML explainability page
    html_path = generate_explainability_html(result)
    console.print(f"\n   Explainability page: [link=file://{html_path}]{html_path}[/link]")
    
    # Open the HTML page
    import webbrowser
    webbrowser.open(f"file://{html_path}")


def cmd_rag(args: list[str]):
    """
    Grounded RAG zoom: Ingest real docs → build hierarchy → serve instant zoom.
    
    The hierarchy is pre-built from REAL document chunks.
    Zooming traces paths from topics down to actual source excerpts.
    Query highlights the relevant paths — the traversal IS the explanation.
    """
    import asyncio
    import pickle
    from gem3.ingest import ingest_documents, count_nodes, count_grounded
    from server import start_server
    
    console.print(Panel.fit(
        "[bold]gem3 Grounded RAG Zoom[/]\n"
        "Real documents → chunked excerpts → Gemini-deduced hierarchy\n"
        "Zoom from topics down to actual source text.\n"
        "[dim]Every leaf is real. Every path is traceable.[/]",
        title="📚 gem3",
        border_style="green",
    ))
    
    # Load or build the knowledge tree
    if KNOWLEDGE_STORE.exists():
        console.print(f"\n  Loading existing knowledge tree from {KNOWLEDGE_STORE}...")
        with open(KNOWLEDGE_STORE, "rb") as f:
            root = pickle.load(f)
        
        # Check if it has positions pre-computed
        if "x" not in root.metadata:
            console.print("  📐 Pre-computing positions...")
            from gem3.ingest import _precompute_positions, CANVAS_WIDTH, CANVAS_HEIGHT
            _precompute_positions(root, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)
    else:
        # Use demo documents
        console.print("\n  No knowledge tree found. Building from demo documents...")
        
        documents = [
            {"title": "Photosynthesis: Light to Life", "content": """Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose. This occurs in chloroplasts within thylakoid structures.\n\nLight-Dependent Reactions occur in thylakoid membranes. Chlorophyll absorbs sunlight, energizing electrons through an electron transport chain. This splits water molecules, releasing oxygen. Energy produces ATP and NADPH.\n\nThe Calvin Cycle uses ATP and NADPH to fix carbon dioxide into organic molecules. The enzyme RuBisCO catalyzes the first step, combining CO2 with RuBP.\n\nThe overall equation: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2\n\nEnvironmental factors include light intensity, CO2 concentration, temperature, and water availability. C4 and CAM plants minimize photorespiration in hot dry environments."""},
            {"title": "Quantum Entanglement", "content": """Quantum entanglement is a phenomenon where particles become correlated so the quantum state of each cannot be described independently. Measuring one instantly affects the other regardless of distance.\n\nWhen two particles interact and become entangled, their quantum states become linked. Two entangled electrons in a singlet state: measuring one as spin-up means the other is always spin-down.\n\nBell's Theorem (1964): John Bell proved no theory of local hidden variables can reproduce quantum mechanics predictions. Aspect's experiments (1982) confirmed this.\n\nApplications include quantum computing with entangled qubits, quantum cryptography using QKD for unbreakable encryption, and quantum teleportation transferring quantum states.\n\nThe EPR Paradox: Einstein, Podolsky, Rosen argued quantum mechanics must be incomplete. However, entanglement doesn't enable faster-than-light communication."""},
            {"title": "CRISPR-Cas9 Gene Editing", "content": """CRISPR-Cas9 is a gene-editing technology adapted from bacterial defense systems. It allows precise modification of DNA sequences in living organisms.\n\nGuide RNA (gRNA) is a short RNA sequence designed to match the target DNA. It directs the Cas9 protein to the right location in the genome.\n\nCas9 Protein acts as molecular scissors, creating a double-strand break in DNA at the target site. The cell's repair mechanisms then kick in.\n\nDNA Repair uses either NHEJ (error-prone, disrupts genes) or HDR (uses template for precise edits).\n\nMedical applications include sickle cell disease treatment via BCL11A gene editing, cancer immunotherapy with engineered T-cells, and CRISPR-based diagnostics like SHERLOCK."""},
            {"title": "Neural Networks and Deep Learning", "content": """Neural networks are computational systems inspired by biological neural networks. They consist of interconnected nodes organized in layers.\n\nArchitecture includes input layers receiving raw data, hidden layers transforming data through weighted connections, and output layers producing predictions.\n\nBackpropagation calculates the gradient of loss with respect to each weight, adjusting weights to minimize error using the chain rule of calculus.\n\nActivation functions include ReLU (max(0,x)), Sigmoid (1/(1+e^-x)), and Tanh. These determine whether neurons should fire.\n\nModern architectures include CNNs for image processing, RNNs for sequential data, and Transformers using self-attention mechanisms for NLP tasks like GPT and BERT."""},
            {"title": "Black Holes and Spacetime", "content": """A black hole is a region where gravity is so intense nothing can escape past the event horizon. They form when massive stars collapse.\n\nTypes include stellar (5-100 solar masses), supermassive (millions to billions, at galaxy centers like Sagittarius A*), intermediate, and primordial.\n\nSchwarzschild Radius r_s = 2GM/c² defines the event horizon for non-rotating black holes. For the Sun this would be about 3 km.\n\nHawking Radiation: Black holes slowly emit radiation due to quantum effects near the event horizon. Virtual particle pairs form; one escapes as real radiation.\n\nThe Information Paradox asks whether black holes destroy information, violating quantum unitarity. The Event Horizon Telescope captured the first black hole image (M87*) in 2019."""},
        ]
        
        with console.status("[bold green]Ingesting documents and building grounded hierarchy..."):
            root = ingest_documents(documents)
        
        with open(KNOWLEDGE_STORE, "wb") as f:
            pickle.dump(root, f)
    
    # Print stats
    counts = count_nodes(root)
    n_grounded = count_grounded(root)
    
    console.print(f"\n  [green]✓[/] Knowledge tree ready:")
    for level, count in sorted(counts.items(), key=lambda x: ["library","topic","subtopic","chapter","section","concept","excerpt"].index(x[0]) if x[0] in ["library","topic","subtopic","chapter","section","concept","excerpt"] else 99):
        console.print(f"    {level}: {count}")
    console.print(f"    [bold green]grounded excerpts: {n_grounded}[/]")
    
    console.print(f"\n  Starting zoom server...")
    asyncio.run(start_server(tree=root))


def cmd_bq(args: list[str]):
    """
    BigQuery ingest: Pull knowledge from BigQuery → build zoom → upload to GCS.
    
    Usage:
      python main.py bq pharmacy           Pull digital pharmacy content (126 docs)
      python main.py bq chicken            Pull chicken restaurant content (1728 docs)
      python main.py bq wiki "quantum"     Pull Wikidata entities matching topic
      python main.py bq upload             Upload outputs to GCS
    """
    import asyncio
    import pickle
    from gem3.bq_ingest import (
        pull_content_normalization, pull_wikidata,
        upload_outputs_to_gcs,
    )
    from gem3.ingest import ingest_documents, count_nodes, count_grounded
    
    if not args:
        console.print("[red]Usage:[/] python main.py bq <pharmacy|chicken|wiki|upload>")
        return
    
    subcmd = args[0].lower()
    
    if subcmd == "upload":
        console.print(Panel.fit(
            "[bold]Uploading outputs to GCS...[/]",
            title="☁️  gem3 GCS Upload",
            border_style="cyan",
        ))
        urls = upload_outputs_to_gcs()
        for name, url in urls.items():
            console.print(f"  [green]✓[/] {name}: [link={url}]{url}[/link]")
        return
    
    console.print(Panel.fit(
        f"[bold]gem3 BigQuery Ingest[/]\n"
        f"Pull documents from BigQuery → build knowledge tree → serve zoom\n"
        f"[dim]Source: {subcmd}[/]",
        title=" gem3 × BigQuery",
        border_style="cyan",
    ))
    
    # Pull documents based on source
    if subcmd == "pharmacy":
        documents = pull_content_normalization("digital_pharmacy_content")
    elif subcmd == "chicken":
        documents = pull_content_normalization("chicken_restaurant_content")
    elif subcmd == "wiki":
        topic = " ".join(args[1:]) if len(args) > 1 else "quantum"
        documents = pull_wikidata(topic_filter=topic)
    else:
        console.print(f"[red]Unknown source:[/] {subcmd}")
        return
    
    if not documents:
        console.print("[red]No documents found.[/]")
        return
    
    console.print(f"\n  📚 {len(documents)} documents pulled. Building knowledge tree...\n")
    
    # Ingest into gem3 hierarchy
    with console.status("[bold green]Building knowledge hierarchy with Gemini..."):
        root = ingest_documents(documents)
    
    # Save
    with open(KNOWLEDGE_STORE, "wb") as f:
        pickle.dump(root, f)
    
    counts = count_nodes(root)
    n_grounded = count_grounded(root)
    
    console.print(f"\n  [green]✓[/] Knowledge tree built:")
    for level, count in sorted(counts.items()):
        console.print(f"    {level}: {count}")
    console.print(f"    [bold green]grounded excerpts: {n_grounded}[/]")
    
    # Upload tree to GCS
    console.print(f"\n  ☁️  Uploading to GCS...")
    from gem3.bq_ingest import upload_to_gcs
    upload_to_gcs(KNOWLEDGE_STORE, "trees")
    
    console.print(f"\n  [bold green]Done![/] Run 'python main.py rag' to serve, or 'python main.py export' to export HTML.")


def cmd_export(args: list[str]):
    """
    Export as a standalone HTML file — host anywhere with zero cost.
    
    The grounded zoom mode is 100% client-side after page load,
    so it can be a single HTML file on GitHub Pages, Netlify, S3, etc.
    """
    import pickle
    from gem3.ingest import count_nodes, count_grounded
    from server import build_grounded_html, find_relevant_paths
    
    query = " ".join(args) if args else ""
    
    if not KNOWLEDGE_STORE.exists():
        console.print("[red]Error:[/] No knowledge tree found. Run 'demo' or 'ingest' first.")
        sys.exit(1)
    
    with open(KNOWLEDGE_STORE, "rb") as f:
        root = pickle.load(f)
    
    if "x" not in root.metadata:
        from gem3.ingest import _precompute_positions, CANVAS_WIDTH, CANVAS_HEIGHT
        _precompute_positions(root, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)
    
    # Find highlighted paths if query provided
    highlighted = set()
    if query:
        highlighted = find_relevant_paths(root, query)
    
    # Generate the complete standalone HTML
    html = build_grounded_html(root, query, highlighted)
    
    # Write to output
    export_path = OUTPUT_DIR / "gem3_export.html"
    with open(export_path, "w") as f:
        f.write(html)
    
    counts = count_nodes(root)
    n_grounded = count_grounded(root)
    
    console.print(Panel.fit(
        f"[bold green]✓ Exported standalone HTML[/]\n\n"
        f"  File: {export_path}\n"
        f"  Size: {export_path.stat().st_size / 1024:.0f} KB\n"
        f"  Nodes: {sum(counts.values())} ({n_grounded} grounded excerpts)\n"
        f"  Query: {query or '(none — all nodes shown)'}\n\n"
        f"[dim]This file is 100% self-contained. Host it anywhere:[/]\n"
        f"  • GitHub Pages: push to gh-pages branch\n"
        f"  • Netlify/Vercel: drag & drop\n"
        f"  • S3/GCS: upload as static file\n"
        f"  • Or just open locally in a browser",
        title=" gem3 Export",
        border_style="green",
    ))
    
    import webbrowser
    webbrowser.open(f"file://{export_path.resolve()}")


def main():
    if len(sys.argv) < 2:
        console.print(__doc__)
        sys.exit(0)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "search" or cmd == "s":
        cmd_search(sys.argv[2:])
    elif cmd == "bq":
        cmd_bq(sys.argv[2:])
    elif cmd == "export" or cmd == "e":
        cmd_export(sys.argv[2:])
    elif cmd == "rag" or cmd == "r":
        cmd_rag(sys.argv[2:])
    elif cmd == "zoom" or cmd == "z":
        cmd_zoom(sys.argv[2:])
    elif cmd == "ingest" and len(sys.argv) >= 3:
        cmd_ingest(sys.argv[2])
    elif cmd == "query" and len(sys.argv) >= 3:
        cmd_query(" ".join(sys.argv[2:]))
    elif cmd == "demo":
        cmd_demo()
    elif cmd == "interactive" or cmd == "i":
        cmd_interactive()
    else:
        console.print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
