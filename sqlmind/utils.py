import hashlib
import os
import uuid
import re
from typing import Dict, List, Any, Union

import pandas as pd


def deterministic_uuid(content: Union[str, bytes]) -> str:
    """
    Generate a deterministic UUID based on content hash.
    
    Args:
        content: String or bytes to hash
        
    Returns:
        UUID string
    """
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise ValueError(f"Content type {type(content)} not supported")
        
    hash_object = hashlib.sha256(content_bytes)
    hash_hex = hash_object.hexdigest()
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    content_uuid = str(uuid.uuid5(namespace, hash_hex))
    
    return content_uuid


def extract_tables_from_sql(sql: str) -> List[str]:
    """
    Extract table names from a SQL query.
    
    Args:
        sql: SQL query
        
    Returns:
        List of table names
    """
    # This is a simplified version - a production version would need more robust parsing
    from_pattern = r"FROM\s+([a-zA-Z0-9_\.]+)"
    join_pattern = r"JOIN\s+([a-zA-Z0-9_\.]+)"
    
    from_tables = re.findall(from_pattern, sql, re.IGNORECASE)
    join_tables = re.findall(join_pattern, sql, re.IGNORECASE)
    
    tables = from_tables + join_tables
    
    # Clean up schema prefixes
    clean_tables = []
    for table in tables:
        parts = table.split(".")
        clean_tables.append(parts[-1])
        
    return list(set(clean_tables))


def extract_column_info_from_schema(schema: str) -> Dict[str, List[str]]:
    """
    Extract column information from schema DDL.
    
    Args:
        schema: SQL DDL schema
        
    Returns:
        Dictionary mapping table names to their columns
    """
    create_table_pattern = r"CREATE\s+TABLE\s+([a-zA-Z0-9_\.]+)\s*\((.*?)\)"
    column_pattern = r"([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+)"
    
    table_info = {}
    matches = re.findall(create_table_pattern, schema, re.IGNORECASE | re.DOTALL)
    
    for table_name, columns_text in matches:
        table_name = table_name.split(".")[-1]  # Remove schema prefix if present
        
        columns = re.findall(column_pattern, columns_text)
        column_names = [col[0] for col in columns]
        
        table_info[table_name] = column_names
        
    return table_info


def extract_python_code(markdown_string: str) -> str:
    """
    Extract Python code from markdown string.
    
    Args:
        markdown_string: String potentially containing code blocks
        
    Returns:
        Extracted Python code
    """
    # Strip whitespace to avoid indentation errors
    markdown_string = markdown_string.strip()
    
    # Regex pattern to match Python code blocks
    pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"
    
    # Find matches
    matches = re.findall(pattern, markdown_string, re.IGNORECASE)
    
    # Extract code
    python_code = []
    for match in matches:
        code = match[0] if match[0] else match[1]
        python_code.append(code.strip())
        
    if not python_code:
        return markdown_string
        
    return python_code[0]


def parse_csv_to_df(file_path: str) -> pd.DataFrame:
    """
    Parse a CSV file into a pandas DataFrame with robust error handling.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame
    """
    # Try different parameters to handle various CSV formats
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"First attempt failed: {e}")
        try:
            return pd.read_csv(file_path, sep=";")
        except Exception as e:
            print(f"Second attempt failed: {e}")
            try:
                return pd.read_csv(file_path, encoding="latin1")
            except Exception as e:
                print(f"Third attempt failed: {e}")
                try:
                    import csv
                    return pd.read_csv(file_path, dialect=csv.excel_tab)
                except Exception as e:
                    raise ValueError(f"Failed to parse CSV: {e}")


def validate_config_path(path: str) -> bool:
    """
    Validate that a configuration file path exists and is readable.
    
    Args:
        path: File path to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError if path is invalid
    """
    if not os.path.exists(path):
        raise ValueError(f"Config file not found: {path}")
    
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")
    
    if not os.access(path, os.R_OK):
        raise ValueError(f"Cannot read config file: {path}")
    
    return True


def sql_to_pandas(sql: str) -> str:
    """
    Convert SQL query to equivalent pandas operations.
    
    Args:
        sql: SQL query
        
    Returns:
        Python code using pandas
    """
    # This is a placeholder - a full implementation would need SQL parsing
    # and conversion to pandas operations, which is quite complex
    return f"""
# Equivalent pandas code for:
# {sql}
# Note: This is a simplified representation

# Assuming tables are already loaded as DataFrames
# df = run_sql("{sql}")

# For a proper implementation, use libraries like pandasql:
# from pandasql import sqldf
# df = sqldf('''{sql}''', globals())
"""

def extract_content_from_xml_tags(text: str, tag_name: str) -> str:
    """
    Extract content from XML tags.
    
    Args:
        text: Text containing XML tags
        tag_name: Name of the XML tag to extract content from
        
    Returns:
        Content within the XML tags
    """
    import re
    
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[-1].strip()
    
    return None


def format_sql_with_xml_tags(sql: str) -> str:
    """
    Formats SQL query with XML tags.
    
    Args:
        sql: SQL query
        
    Returns:
        SQL query wrapped in <sql> tags
    """
    formatted_sql = sql.strip()
    
    # Only wrap if not already wrapped
    if not formatted_sql.lower().startswith("<sql>") and not formatted_sql.lower().endswith("</sql>"):
        formatted_sql = f"<sql>\n{formatted_sql}\n</sql>"
        
    return formatted_sql