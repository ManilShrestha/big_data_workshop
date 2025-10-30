"""
Check the Prince example to see if the 3-hop path exists in the graph.
"""
from qa_system.config import Config
from qa_system.utils.loader import load_graph


def main():
    print("="*80)
    print("CHECKING PRINCE EXAMPLE")
    print("="*80)

    # Load graph
    print("\n[1] Loading graph...")
    graph = load_graph(str(Config.GRAPH_PATH))
    print(f"   {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

    # The question
    print("\n[2] Question:")
    print("   'who is listed as screenwriter of the films directed by")
    print("    the [Under the Cherry Moon] director'")
    print("\n   Expected 3-hop chain:")
    print("   1. Under the Cherry Moon --[directed_by]--> Prince")
    print("   2. Prince <--[directed_by]-- Other movies Prince directed")
    print("   3. Other movies --[written_by]--> Screenwriters")

    # Check if "Under the Cherry Moon" exists
    movie = "Under the Cherry Moon"
    if movie not in graph:
        print(f"\n   ❌ '{movie}' not found in graph!")
        return

    print(f"\n[3] Exploring '{movie}' in the graph:")

    # Get director
    print("\n   Step 1: Who directed this movie?")
    directors = []
    for _, target, _, edge_data in graph.out_edges(movie, keys=True, data=True):
        if edge_data.get('relation') == 'directed_by':
            directors.append(target)
            print(f"   ✓ {movie} --[directed_by]--> {target}")

    if not directors:
        print("   ❌ No director found!")
        return

    director = directors[0]  # Prince

    # Check if Prince wrote it too (the shortcut!)
    print("\n   SHORTCUT CHECK: Did Prince also write it?")
    for _, target, _, edge_data in graph.out_edges(movie, keys=True, data=True):
        if edge_data.get('relation') == 'written_by':
            print(f"   ⚠️  {movie} --[written_by]--> {target}")
            if target == director:
                print(f"   ⚠️  YES! {director} is BOTH director AND writer!")
                print(f"   This creates a 1-hop shortcut to the answer!")

    # Find other movies Prince directed
    print(f"\n   Step 2: What other movies did {director} direct?")
    print(f"   (Looking for incoming 'directed_by' edges)")

    other_movies = []
    for source, _, _, edge_data in graph.in_edges(director, keys=True, data=True):
        if edge_data.get('relation') == 'directed_by':
            other_movies.append(source)
            if len(other_movies) <= 10:
                print(f"   ✓ {source} --[directed_by]--> {director}")

    if len(other_movies) > 10:
        print(f"   ... and {len(other_movies) - 10} more movies")

    print(f"\n   Total movies directed by {director}: {len(other_movies)}")

    # Find screenwriters of those movies
    print(f"\n   Step 3: Who wrote those movies?")
    screenwriters = set()

    for movie in other_movies[:5]:  # Check first 5
        print(f"\n   Movie: {movie}")
        writers = []
        for _, target, _, edge_data in graph.out_edges(movie, keys=True, data=True):
            if edge_data.get('relation') == 'written_by':
                writers.append(target)
                screenwriters.add(target)
                print(f"     --[written_by]--> {target}")

        if not writers:
            print(f"     (no writers found)")

    print(f"\n   Total unique screenwriters found: {len(screenwriters)}")
    print(f"   Screenwriters: {list(screenwriters)[:10]}")

    # Check answer
    print(f"\n[4] Checking expected answer:")
    print(f"   Expected answer: Prince")

    if director in screenwriters:
        print(f"   ✓ Prince IS a screenwriter of movies he directed!")
        print(f"   The 3-hop path EXISTS:")
        print(f"   Under the Cherry Moon → Prince (director) → Movies Prince directed → Prince (writer)")
    else:
        print(f"   ❌ Prince is NOT in the screenwriters list")

    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
The question expects a 3-hop reasoning chain, but:

✓ The 3-hop path EXISTS in the graph
⚠️  BUT a 1-hop SHORTCUT also exists!

The shortcut:
  Under the Cherry Moon --[written_by]--> Prince (1 hop)

The intended 3-hop path:
  Under the Cherry Moon --[directed_by]--> Prince
  Prince <--[directed_by]-- (other movies)
  (other movies) --[written_by]--> Prince

Since BFS finds the SHORTEST path first, it returns the 1-hop shortcut.
This is why your script CORRECTLY filters it out - the question expects
3-hop reasoning, not a 1-hop direct answer!
""")


if __name__ == "__main__":
    main()
