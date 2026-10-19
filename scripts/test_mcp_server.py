import asyncio
import sys

from mcp import (
    ClientSession,
    StdioServerParameters,
)
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "app.mcp_server.server",
        ],
    )

    async with stdio_client(server_params) as (
        read,
        write,
    ):
        async with ClientSession(
            read,
            write,
        ) as session:
            await session.initialize()

            print("=" * 60)
            print("TOOLS")
            print("=" * 60)

            tools = await session.list_tools()

            for tool in tools.tools:
                print(tool.name)

            print()
            print("=" * 60)
            print("DOCUMENT COUNT")
            print("=" * 60)

            result = await session.call_tool(
                "get_document_count",
                arguments={},
            )

            print(result.content)

            print()
            print("=" * 60)
            print("SEARCH")
            print("=" * 60)

            result = await session.call_tool(
                "search_knowledge",
                arguments={
                    "query": ("Quel outil exécute le LLM localement ?"),
                    "top_k": 3,
                },
            )
            print()

            print("=" * 60)
            print("CATEGORY COUNTS")
            print("=" * 60)

            result = await session.call_tool(
                "get_category_counts",
                arguments={},
            )

            print(result.content)

            print()
            print("=" * 60)
            print("RECENT DOCUMENTS")
            print("=" * 60)

            result = await session.call_tool(
                "get_recent_documents",
                arguments={
                    "limit": 3,
                },
            )

            print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
