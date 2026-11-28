"""
Quick local test — run this to verify the agent works before pushing.

Usage:
    python test_agent.py
"""

from agent import TavilySearchAgent


def main():
    idea = "AI-powered resume screener for HR teams"
    print(f"\n🔍 Running market research for: '{idea}'\n")
    print("=" * 60)

    agent = TavilySearchAgent()
    output = agent.run(idea, max_results=3)

    print(f"Startup Idea : {output['idea']}")
    print(f"Queries Run  : {len(output['queries'])}")
    print()

    for query, results in output["results"].items():
        print(f"📌 Query: {query}")
        for r in results:
            if "error" in r:
                print(f"   ❌ Error: {r['error']}")
            else:
                print(f"   • {r['title']}")
                print(f"     {r['url']}")
        print()

    print("📊 Summary by category:")
    for category, text in output["summary"].items():
        print(f"\n  [{category}]")
        print(f"  {text[:200]}...")


if __name__ == "__main__":
    main()
