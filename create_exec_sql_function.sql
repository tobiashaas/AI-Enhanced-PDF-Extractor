-- Diese SQL-Funktion muss in Supabase erstellt werden, 
-- damit das fix_images_table.py Skript funktioniert

-- Funktion zum Ausführen von beliebigen SQL-Befehlen
-- ACHTUNG: Nur für den Admin-Gebrauch!
CREATE OR REPLACE FUNCTION exec_sql(query text)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  EXECUTE query;
END;
$$;

-- Zugriff nur für den service_role
REVOKE ALL ON FUNCTION exec_sql(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION exec_sql(text) TO service_role;