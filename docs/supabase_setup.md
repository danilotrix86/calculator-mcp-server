# Supabase Integration Setup

This document explains how to set up Supabase integration with the calculator-mcp-server.

## 1. Create a Supabase Account and Project

1. Go to [https://supabase.com/](https://supabase.com/) and create an account or log in
2. Create a new project
3. Note your Supabase URL and API Key (you'll need both the anon key and service role key)

## 2. Create the Database Table

Run the following SQL in the Supabase SQL Editor:

```sql
CREATE TABLE user_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question TEXT NOT NULL,
    response TEXT,
    tool_used TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index on created_at for faster queries
CREATE INDEX user_queries_created_at_idx ON user_queries(created_at);
```

## 3. Set Environment Variables

Add these variables to your environment:

```
# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
```

You can add them to your `.env` file, or set them in your deployment environment.

## 4. Usage

The integration is already implemented in the solve routes. When users make requests:

1. The query is saved to Supabase immediately
2. When the response is ready, it's updated in the same record
3. Errors are also tracked

## 5. Query Your Data

You can use Supabase's interface to query your data or connect to the database directly.

Example query to see recent queries:

```sql
SELECT 
    id, 
    question, 
    response, 
    tool_used, 
    created_at 
FROM 
    user_queries 
ORDER BY 
    created_at DESC 
LIMIT 100;
```

## 6. Monitoring

Consider setting up Supabase Edge Functions to create alerts or reports based on the data.
