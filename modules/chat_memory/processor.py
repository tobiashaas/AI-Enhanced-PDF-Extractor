#!/usr/bin/env python3
"""
Chat Memory Module
----------------
Modul zur Verwaltung des Chat-Gedächtnisses und Kontextualisierung für den AI-Agenten.
Speichert und verwaltet technische Details, Projekt-Entscheidungen und Anwendungswissen.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class MemoryManager:
    """Verwaltet das Gedächtnis des AI-Agenten für Chats und technische Details."""
    
    def __init__(self, supabase_client, config):
        """
        Initialisiert den Memory Manager
        
        Args:
            supabase_client: Client für die Supabase-Verbindung
            config: Konfigurationsobjekt
        """
        self.supabase = supabase_client
        self.config = config
        self.memory_dir = Path(config.get("memory_dir", "MEMORY"))
        self.memory_cache = {}
        self.technical_cheat_sheet = {}
        self.project_master_plan = {}
        self._load_memory_files()
        
        logger.info("Memory Manager initialisiert")

    def _load_memory_files(self) -> None:
        """Lädt Memory-Dateien aus dem MEMORY-Verzeichnis"""
        # Technisches Cheat Sheet laden
        cheat_sheet_path = self.memory_dir / "TECHNICAL_CHEAT_SHEET.md"
        if cheat_sheet_path.exists():
            self.technical_cheat_sheet = self._parse_markdown_to_dict(cheat_sheet_path)
            logger.info("Technisches Cheat Sheet geladen")
        else:
            logger.warning(f"Technisches Cheat Sheet nicht gefunden: {cheat_sheet_path}")
        
        # Projekt-Masterplan laden
        plan_path = self.memory_dir / "PROJECT_MASTER_PLAN.md"
        if plan_path.exists():
            self.project_master_plan = self._parse_markdown_to_dict(plan_path)
            logger.info("Projekt-Masterplan geladen")
        else:
            logger.warning(f"Projekt-Masterplan nicht gefunden: {plan_path}")

    def _parse_markdown_to_dict(self, file_path: Path) -> Dict[str, Any]:
        """
        Parst eine Markdown-Datei in ein strukturiertes Dictionary
        
        Args:
            file_path: Pfad zur Markdown-Datei
            
        Returns:
            Dict[str, Any]: Strukturiertes Dictionary aus dem Markdown-Inhalt
        """
        result = {}
        current_section = None
        current_subsection = None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    
                    # Überschriften verarbeiten
                    if line.startswith('# '):
                        current_section = line[2:].strip()
                        result[current_section] = {}
                        current_subsection = None
                    elif line.startswith('## ') and current_section:
                        current_subsection = line[3:].strip()
                        result[current_section][current_subsection] = []
                    elif line.startswith('- ') and current_section:
                        if current_subsection:
                            result[current_section][current_subsection].append(line[2:].strip())
                        else:
                            if 'items' not in result[current_section]:
                                result[current_section]['items'] = []
                            result[current_section]['items'].append(line[2:].strip())
                    elif line and current_section and current_subsection and not line.startswith('#'):
                        # Ergänzt bestehenden Inhalt oder fügt neuen hinzu
                        result[current_section][current_subsection].append(line)
        except Exception as e:
            logger.error(f"Fehler beim Parsen der Markdown-Datei {file_path}: {e}")
        
        return result

    def get_technical_info(self, topic: str = None) -> Dict[str, Any]:
        """
        Gibt technische Informationen aus dem Cheat Sheet zurück
        
        Args:
            topic: Optionaler Topic-Filter
            
        Returns:
            Dict[str, Any]: Technische Informationen
        """
        if not topic:
            return self.technical_cheat_sheet
            
        result = {}
        for section, content in self.technical_cheat_sheet.items():
            if topic.lower() in section.lower():
                result[section] = content
                
        return result

    def get_project_plan(self, section: str = None) -> Dict[str, Any]:
        """
        Gibt den Projekt-Masterplan zurück
        
        Args:
            section: Optionaler Abschnittsfilter
            
        Returns:
            Dict[str, Any]: Projekt-Masterplan
        """
        if not section:
            return self.project_master_plan
            
        result = {}
        for plan_section, content in self.project_master_plan.items():
            if section.lower() in plan_section.lower():
                result[plan_section] = content
                
        return result

    def record_chat_session(self, session_data: Dict[str, Any]) -> str:
        """
        Zeichnet eine Chat-Sitzung in der Datenbank auf
        
        Args:
            session_data: Daten der Chat-Sitzung
            
        Returns:
            str: ID der Chat-Sitzung
        """
        try:
            # Chat-Session in Supabase speichern
            result = self.supabase.table("chat_sessions").insert({
                "session_id": session_data.get("session_id", hashlib.md5(str(time.time()).encode()).hexdigest()),
                "summary": session_data.get("summary", ""),
                "context": session_data.get("context", {}),
                "key_decisions": session_data.get("key_decisions", []),
                "timestamp": datetime.now().isoformat(),
                "metadata": session_data.get("metadata", {})
            }).execute()
            
            if result.data:
                session_id = result.data[0]["id"]
                # Chat-Nachrichten speichern
                if "messages" in session_data:
                    for msg in session_data["messages"]:
                        self.supabase.table("chat_messages").insert({
                            "session_id": session_id,
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                            "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                            "metadata": msg.get("metadata", {})
                        }).execute()
                
                return session_id
        except Exception as e:
            logger.error(f"Fehler beim Aufzeichnen der Chat-Sitzung: {e}")
        
        return None

    def retrieve_relevant_context(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Ruft relevanten Kontext für eine Benutzeranfrage ab
        
        Args:
            query: Benutzeranfrage
            max_results: Maximale Anzahl von Ergebnissen
            
        Returns:
            List[Dict[str, Any]]: Relevanter Kontext
        """
        try:
            # Einfache Keyword-basierte Suche in Chat-Sessions
            result = self.supabase.table("chat_sessions") \
                       .select("id, summary, key_decisions, timestamp") \
                       .ilike("summary", f"%{query}%") \
                       .limit(max_results) \
                       .execute()
                       
            # Technisches Cheat Sheet durchsuchen
            tech_info = {}
            for topic, info in self.technical_cheat_sheet.items():
                if query.lower() in topic.lower():
                    tech_info[topic] = info
            
            # Projekt-Plan durchsuchen
            plan_info = {}
            for section, content in self.project_master_plan.items():
                if query.lower() in section.lower():
                    plan_info[section] = content
            
            # Ergebnisse zusammenführen
            context = []
            
            if result.data:
                for session in result.data:
                    context.append({
                        "type": "chat_session",
                        "id": session["id"],
                        "summary": session["summary"],
                        "key_decisions": session["key_decisions"],
                        "timestamp": session["timestamp"]
                    })
            
            if tech_info:
                context.append({
                    "type": "technical_info",
                    "data": tech_info
                })
                
            if plan_info:
                context.append({
                    "type": "project_plan",
                    "data": plan_info
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen von relevantem Kontext: {e}")
            return []

    def update_technical_cheat_sheet(self, updates: Dict[str, Any]) -> bool:
        """
        Aktualisiert das technische Cheat Sheet
        
        Args:
            updates: Aktualisierungen für das Cheat Sheet
            
        Returns:
            bool: True, wenn erfolgreich
        """
        try:
            # Bestehende Informationen mit Updates zusammenführen
            for section, content in updates.items():
                if section in self.technical_cheat_sheet:
                    # Bestehenden Abschnitt aktualisieren
                    if isinstance(content, dict):
                        for subsection, items in content.items():
                            self.technical_cheat_sheet[section][subsection] = items
                    else:
                        self.technical_cheat_sheet[section] = content
                else:
                    # Neuen Abschnitt hinzufügen
                    self.technical_cheat_sheet[section] = content
            
            # Aktualisierte Informationen in die Datei schreiben
            self._write_dict_to_markdown(
                self.technical_cheat_sheet,
                self.memory_dir / "TECHNICAL_CHEAT_SHEET.md"
            )
            
            return True
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren des technischen Cheat Sheets: {e}")
            return False

    def update_project_master_plan(self, updates: Dict[str, Any]) -> bool:
        """
        Aktualisiert den Projekt-Masterplan
        
        Args:
            updates: Aktualisierungen für den Masterplan
            
        Returns:
            bool: True, wenn erfolgreich
        """
        try:
            # Bestehende Informationen mit Updates zusammenführen
            for section, content in updates.items():
                if section in self.project_master_plan:
                    # Bestehenden Abschnitt aktualisieren
                    if isinstance(content, dict):
                        for subsection, items in content.items():
                            self.project_master_plan[section][subsection] = items
                    else:
                        self.project_master_plan[section] = content
                else:
                    # Neuen Abschnitt hinzufügen
                    self.project_master_plan[section] = content
            
            # Aktualisierte Informationen in die Datei schreiben
            self._write_dict_to_markdown(
                self.project_master_plan, 
                self.memory_dir / "PROJECT_MASTER_PLAN.md"
            )
            
            return True
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren des Projekt-Masterplans: {e}")
            return False

    def _write_dict_to_markdown(self, data: Dict[str, Any], file_path: Path) -> None:
        """
        Schreibt ein Dictionary in eine Markdown-Datei
        
        Args:
            data: Zu schreibende Daten
            file_path: Pfad zur Markdown-Datei
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            for section, content in data.items():
                file.write(f"# {section}\n\n")
                
                if isinstance(content, dict):
                    for subsection, items in content.items():
                        file.write(f"## {subsection}\n\n")
                        
                        if isinstance(items, list):
                            for item in items:
                                file.write(f"- {item}\n")
                        else:
                            file.write(f"{items}\n")
                            
                        file.write("\n")
                elif isinstance(content, list):
                    for item in content:
                        file.write(f"- {item}\n")
                    file.write("\n")
                else:
                    file.write(f"{content}\n\n")