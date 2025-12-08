"""
Command Line Interface for Legal Assistant RAG Chatbot.
"""

import sys
import asyncio
import rich
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import Config
from src.retriever import get_retriever
from src.llm_client import get_llm_client
from src.prompt_templates import (
    build_research_prompt,
    build_judgment_prompt,
    build_summarize_prompt
)
from src.verify_and_log import (
    verify_bracket_citations,
    create_log_entry,
    write_log_file,
    should_retry_generation,
    build_retry_prompt
)

console = Console()

def print_header():
    console.print(Panel.fit(
        "[bold blue]Legal Assistant RAG Chatbot[/bold blue]\n[italic]AI Legal Assistant for Indian Law[/italic]",
        border_style="blue"
    ))

def display_menu():
    console.print("\n[bold]Select Mode:[/bold]")
    console.print("1. [cyan]Research[/cyan] (Ask legal questions)")
    console.print("2. [cyan]Judgment[/cyan] (Simulate or reference judgments)")
    console.print("3. [cyan]Summarize[/cyan] (Generate headnotes/summaries)")
    console.print("4. [red]Exit[/red]")

async def handle_research():
    console.print("\n[bold cyan]-- Research Mode --[/bold cyan]")
    query = Prompt.ask("Enter your legal query")
    
    if not query:
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description="Retrieving documents...", total=None)
        
        # Retrieve
        retriever = get_retriever()
        retrieved = retriever.retrieve(query, top_k=Config.DEFAULT_TOP_K)
        
        if not retrieved:
            console.print("[red]No relevant documents found.[/red]")
            return

        progress.update(task, description="Generating answer...")
        
        # Generate
        prompt = build_research_prompt(query, retrieved)
        llm_client = get_llm_client()
        result = llm_client.generate(prompt, temperature=Config.RESEARCH_TEMPERATURE)
        answer = result["text"]
        
        # Verify
        progress.update(task, description="Verifying citations...")
        verification = verify_bracket_citations(answer, len(retrieved))
        
        if should_retry_generation(verification):
            progress.update(task, description="Refining answer based on verification...")
            retry_prompt = build_retry_prompt(prompt, len(retrieved), verification["invalid"])
            result = llm_client.generate(retry_prompt, temperature=Config.RESEARCH_TEMPERATURE)
            answer = result["text"]
            verification = verify_bracket_citations(answer, len(retrieved))

    # Display Result
    console.print("\n[bold green]Answer:[/bold green]")
    console.print(Markdown(answer))
    
    # Display Sources
    console.print("\n[bold]Sources:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=4)
    table.add_column("Case/Act", style="cyan")
    table.add_column("Citation/Section")
    
    for i, doc in enumerate(retrieved, 1):
        meta = doc["metadata"]
        name = meta.get("case_name") or meta.get("act_name") or "Unknown"
        citation = meta.get("citation") or f"Section {meta.get('section', '?')}"
        table.add_row(str(i), name, str(citation))
    
    console.print(table)
    
    # Log
    log_entry = create_log_entry(
        mode="research",
        user_input=query,
        retrieved=retrieved,
        prompt=prompt,
        llm_response=answer,
        verification=verification,
        temperature=Config.RESEARCH_TEMPERATURE
    )
    logfile = write_log_file(log_entry, "research")
    console.print(f"\n[dim]Log saved to: {logfile}[/dim]")

async def handle_judgment():
    console.print("\n[bold cyan]-- Judgment Mode --[/bold cyan]")
    facts = Prompt.ask("Enter case facts")
    mode_choice = Prompt.ask("Type (h)ypothetical or (r)eference", choices=["h", "r"], default="h")
    mode = "hypothetical" if mode_choice == "h" else "reference"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description="Retrieving precedents...", total=None)
        
        retriever = get_retriever()
        retrieved = retriever.retrieve(facts, top_k=Config.DEFAULT_TOP_K)
        
        if not retrieved:
            console.print("[red]No relevant documents found.[/red]")
            return

        progress.update(task, description="Drafting judgment analysis...")
        
        prompt = build_judgment_prompt(facts, mode, retrieved)
        llm_client = get_llm_client()
        result = llm_client.generate(prompt, temperature=Config.JUDGMENT_TEMPERATURE)
        answer = result["text"]
        
        # Verify
        verification = verify_bracket_citations(answer, len(retrieved))
        
    console.print("\n[bold green]Judgment Analysis:[/bold green]")
    console.print(Markdown(answer))
    
    if mode == "hypothetical":
        console.print("\n[bold red]DISCLAIMER: HYPOTHETICAL ANALYSIS — NOT LEGAL ADVICE[/bold red]")
    
    # Log
    log_entry = create_log_entry(
        mode="judgment",
        user_input=facts,
        retrieved=retrieved,
        prompt=prompt,
        llm_response=answer,
        verification=verification,
        temperature=Config.JUDGMENT_TEMPERATURE
    )
    write_log_file(log_entry, "judgment")

async def handle_summarize():
    console.print("\n[bold cyan]-- Summarize Mode --[/bold cyan]")
    choice = Prompt.ask("Summarize by (q)uery or (t)ext?", choices=["q", "t"], default="q")
    
    query = None
    case_text = None
    retrieved = []
    
    if choice == "q":
        query = Prompt.ask("Enter topic to summarize")
        with Progress(SpinnerColumn(), TextColumn("Retrieving..."), transient=True) as p:
            p.add_task("", total=None)
            retriever = get_retriever()
            retrieved = retriever.retrieve(query, top_k=3)
    else:
        case_text = Prompt.ask("Paste text to summarize")

    with Progress(SpinnerColumn(), TextColumn("Generating summary..."), transient=True) as p:
        p.add_task("", total=None)
        prompt = build_summarize_prompt(query or "", retrieved, case_text)
        llm_client = get_llm_client()
        result = llm_client.generate(prompt, temperature=Config.SUMMARIZE_TEMPERATURE)
        answer = result["text"]

    console.print("\n[bold green]Summary:[/bold green]")
    console.print(Markdown(answer))
    
    # Log
    log_entry = create_log_entry(
        mode="summarize",
        user_input=query or case_text[:100],
        retrieved=retrieved,
        prompt=prompt,
        llm_response=answer,
        verification={},
        temperature=Config.SUMMARIZE_TEMPERATURE
    )
    write_log_file(log_entry, "summarize")

async def main():
    print_header()
    
    while True:
        display_menu()
        choice = Prompt.ask("Choice", choices=["1", "2", "3", "4"])
        
        if choice == "1":
            await handle_research()
        elif choice == "2":
            await handle_judgment()
        elif choice == "3":
            await handle_summarize()
        elif choice == "4":
            console.print("[yellow]Goodbye![/yellow]")
            break
        
        if not Confirm.ask("\nPerform another action?"):
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
        sys.exit(0)
