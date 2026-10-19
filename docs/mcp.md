# MCP integration

## Objective

The project exposes part of the Enterprise AI Knowledge Assistant
through the Model Context Protocol (MCP).

MCP makes the application's tools discoverable and callable by
compatible AI clients through a standardized protocol.

## Architecture

MCP client
    |
    v
MCP server
    |
    +--> search_knowledge
    |       |
    |       v
    |   Hybrid retrieval
    |       |
    |       v
    |   PostgreSQL + pgvector
    |
    +--> get_document_count
    |
    +--> get_category_counts
    |
    +--> get_recent_documents

## Exposed tools

### search_knowledge

Searches the knowledge base using the project's existing hybrid
retrieval pipeline.

Inputs:

- query
- top_k

Returns:

- document
- page
- chunk
- retrieval score
- content

### get_document_count

Returns the total number of documents stored in PostgreSQL.

### get_category_counts

Returns the number of documents grouped by predicted category.

### get_recent_documents

Returns the most recently inserted documents.

## Implementation

The MCP server does not duplicate the application logic.

It reuses the same Python services already used by FastAPI and
LangGraph.

This keeps the architecture modular:

Application services
-> FastAPI
-> LangGraph agent
-> MCP server

## MCP vs LangGraph

LangGraph and MCP solve different problems.

LangGraph is used internally to orchestrate application workflows and
decide which tools should be called.

MCP exposes tools through a standardized protocol so that external AI
clients can discover and invoke them.

In this project:

- LangGraph orchestrates tools.
- MCP exposes tools.

## Transport

The current implementation uses the MCP stdio transport.

A Python MCP client starts the server as a subprocess, discovers the
available tools and invokes them.

## Validation

The test client successfully:

- connects to the MCP server,
- lists the available tools,
- calls PostgreSQL-backed tools,
- calls the hybrid knowledge search,
- receives MCP responses.

## Security considerations

The MCP server exposes only predefined operations.

It does not expose unrestricted SQL execution.

Database access remains controlled by the application's service layer.