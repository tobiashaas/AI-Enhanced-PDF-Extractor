#!/usr/bin/env python3
"""
Intelligente Techniker-Suche für AI Agent
- Erkennt automatisch Hersteller aus Modellnummer
- Fuzzy Search für Tippfehler
- Kontextuelle Suche über mehrere Modelle
- Lernt aus Techniker-Anfragen
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from supabase import create_client

class TechnicianSearchEngine:
    """Intelligente Suche für Techniker-Support"""
    
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.supabase = create_client(
            config['supabase_url'],
            config['supabase_key']
        )
        
        # Hersteller-Modell-Mappings (wird aus DB geladen und erweitert)
        self.manufacturer_patterns = {
            'HP': {
                'prefixes': ['E', 'X', 'M', 'CP'],
                'patterns': [r'[EXM]\d{3,5}', r'CP\d{4}', r'X\d{3}'],
                'aliases': ['hewlett packard', 'hewlett-packard']
            },
            'Canon': {
                'prefixes': ['IR', 'C'],
                'patterns': [r'IR[A-Z]?\d{3,4}', r'C\d{3,4}'],
                'aliases': ['imagerunner']
            },
            'Konica Minolta': {
                'prefixes': ['BH', 'C'],
                'patterns': [r'BH\d{3,4}', r'bizhub\s?\d{3,4}'],
                'aliases': ['bizhub', 'konica', 'minolta']
            },
            'Brother': {
                'prefixes': ['DCP', 'MFC', 'HL'],
                'patterns': [r'(DCP|MFC|HL)[-\s]?\d{3,4}'],
                'aliases': []
            },
            'Xerox': {
                'prefixes': ['WC', 'VL'],
                'patterns': [r'(WorkCentre|WC)[-\s]?\d{3,4}'],
                'aliases': ['workcentre', 'versalink']
            }
        }
    
    def smart_search(self, query: str) -> Dict:
        """Intelligente Suche mit automatischer Erkennung"""
        print(f"🔍 Suche: '{query}'")
        
        # 1. Bereinige und analysiere Query
        cleaned_query = self._clean_query(query)
        
        # 2. Erkenne Hersteller und Modell
        detected = self._detect_manufacturer_model(cleaned_query)
        
        # 3. Erweitere Suche wenn nötig
        search_results = self._execute_search(detected, cleaned_query)
        
        # 4. Bewerte und sortiere Ergebnisse
        ranked_results = self._rank_results(search_results, detected)
        
        return {
            'query': query,
            'detected': detected,
            'results': ranked_results,
            'suggestions': self._generate_suggestions(detected, ranked_results)
        }
    
    def _clean_query(self, query: str) -> str:
        """Bereinige Suchanfrage"""
        # Entferne Füllwörter
        stop_words = ['der', 'die', 'das', 'error', 'fehler', 'problem', 'code']
        
        cleaned = query.lower()
        for word in stop_words:
            cleaned = re.sub(rf'\b{word}\b', '', cleaned)
        
        # Normalisiere Leerzeichen
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _detect_manufacturer_model(self, query: str) -> Dict:
        """Erkenne Hersteller und Modell aus Query"""
        detected = {
            'manufacturer': None,
            'model': None,
            'confidence': 0.0,
            'method': 'unknown'
        }
        
        # Direkte Hersteller-Erwähnung
        for manufacturer, info in self.manufacturer_patterns.items():
            # Prüfe Haupt-Namen
            if manufacturer.lower() in query.lower():
                detected['manufacturer'] = manufacturer
                detected['confidence'] += 0.3
                detected['method'] = 'direct_mention'
            
            # Prüfe Aliases
            for alias in info['aliases']:
                if alias.lower() in query.lower():
                    detected['manufacturer'] = manufacturer
                    detected['confidence'] += 0.2
                    detected['method'] = 'alias'
        
        # Modell-Pattern-Erkennung
        best_match = None
        best_confidence = 0
        
        for manufacturer, info in self.manufacturer_patterns.items():
            for pattern in info['patterns']:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    match_confidence = 0.5
                    
                    # Wenn Hersteller bereits erkannt, höhere Confidence
                    if detected['manufacturer'] == manufacturer:
                        match_confidence = 0.8
                    elif detected['manufacturer'] is None:
                        detected['manufacturer'] = manufacturer
                        match_confidence = 0.6
                    
                    if match_confidence > best_confidence:
                        best_match = matches[0]
                        best_confidence = match_confidence
                        detected['model'] = best_match.upper()
                        detected['confidence'] = match_confidence
                        detected['method'] = 'pattern_match'
        
        return detected
    
    def _execute_search(self, detected: Dict, query: str) -> List[Dict]:
        """Führe Datenbanksuche aus"""
        results = []
        
        try:
            # Suche-Strategien basierend auf erkannten Informationen
            
            if detected['manufacturer'] and detected['model']:
                # Exakte Hersteller+Modell Suche
                print(f"   🎯 Exakte Suche: {detected['manufacturer']} {detected['model']}")
                exact_results = self._search_exact(detected['manufacturer'], detected['model'])
                results.extend(exact_results)
            
            elif detected['manufacturer']:
                # Nur Hersteller bekannt
                print(f"   🏭 Hersteller-Suche: {detected['manufacturer']}")
                manufacturer_results = self._search_by_manufacturer(detected['manufacturer'], query)
                results.extend(manufacturer_results)
            
            elif detected['model']:
                # Nur Modell bekannt - suche über alle Hersteller
                print(f"   🔧 Modell-Suche: {detected['model']}")
                model_results = self._search_by_model(detected['model'])
                results.extend(model_results)
            
            else:
                # Volltext-Suche
                print(f"   📝 Volltext-Suche")
                fulltext_results = self._search_fulltext(query)
                results.extend(fulltext_results)
            
            # Entferne Duplikate
            seen = set()
            unique_results = []
            for result in results:
                key = (result.get('manufacturer', ''), result.get('model', ''), result.get('content', ''))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            
            return unique_results
            
        except Exception as e:
            print(f"   ❌ Suchfehler: {e}")
            return []
    
    def _search_exact(self, manufacturer: str, model: str) -> List[Dict]:
        """Exakte Suche nach Hersteller und Modell"""
        try:
            # Suche in Chunks mit exakter Übereinstimmung
            result = self.supabase.table('chunks') \
                .select('*') \
                .eq('manufacturer', manufacturer) \
                .ilike('model', f'%{model}%') \
                .execute()
            
            return result.data if result.data else []
        except:
            return []
    
    def _search_by_manufacturer(self, manufacturer: str, query: str) -> List[Dict]:
        """Suche nur nach Hersteller"""
        try:
            result = self.supabase.table('chunks') \
                .select('*') \
                .eq('manufacturer', manufacturer) \
                .or_(f'content.ilike.%{query}%,error_codes.cs.{{{query}}}') \
                .limit(20) \
                .execute()
            
            return result.data if result.data else []
        except:
            return []
    
    def _search_by_model(self, model: str) -> List[Dict]:
        """Suche nur nach Modell (alle Hersteller)"""
        try:
            result = self.supabase.table('chunks') \
                .select('*') \
                .ilike('model', f'%{model}%') \
                .execute()
            
            return result.data if result.data else []
        except:
            return []
    
    def _search_fulltext(self, query: str) -> List[Dict]:
        """Volltext-Suche"""
        try:
            result = self.supabase.table('chunks') \
                .select('*') \
                .ilike('content', f'%{query}%') \
                .limit(15) \
                .execute()
            
            return result.data if result.data else []
        except:
            return []
    
    def _rank_results(self, results: List[Dict], detected: Dict) -> List[Dict]:
        """Bewerte und sortiere Suchergebnisse"""
        for result in results:
            score = 0.0
            
            # Hersteller-Match
            if detected['manufacturer'] and result.get('manufacturer') == detected['manufacturer']:
                score += 0.3
            
            # Modell-Match (fuzzy)
            if detected['model'] and result.get('model'):
                similarity = SequenceMatcher(None, detected['model'], result['model']).ratio()
                score += similarity * 0.4
            
            # Content-Relevanz
            if result.get('chunk_type') in ['error_procedure', 'troubleshooting']:
                score += 0.2
            
            # Error Codes vorhanden
            if result.get('error_codes') and len(result['error_codes']) > 0:
                score += 0.1
            
            result['relevance_score'] = score
        
        # Sortiere nach Relevanz
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    def _generate_suggestions(self, detected: Dict, results: List[Dict]) -> List[str]:
        """Generiere Verbesserungs-Vorschläge"""
        suggestions = []
        
        if not detected['manufacturer'] and results:
            # Vorschlag: Spezifiziere Hersteller
            manufacturers = set(r.get('manufacturer') for r in results[:5] if r.get('manufacturer'))
            if manufacturers:
                suggestions.append(f"💡 Spezifiziere Hersteller: {', '.join(manufacturers)}")
        
        if not detected['model'] and results:
            # Vorschlag: Spezifiziere Modell
            models = set(r.get('model') for r in results[:5] if r.get('model'))
            if models:
                suggestions.append(f"💡 Spezifiziere Modell: {', '.join(list(models)[:3])}")
        
        if len(results) == 0:
            suggestions.append("💡 Versuche: Modellnummer, Fehlernummer oder Symptom")
            suggestions.append("💡 Beispiele: 'E52645 C0001', 'HP Paper Jam', 'Scanner Error'")
        
        return suggestions

def demo_search():
    """Demo der intelligenten Suche"""
    engine = TechnicianSearchEngine()
    
    test_queries = [
        "E52645",           # Nur Modell
        "HP E52645",        # Hersteller + Modell  
        "E52645 C0001",     # Modell + Error Code
        "Paper Jam",        # Nur Symptom
        "HP Scanner Error", # Hersteller + Problem
        "Canon IR3300",     # Anderer Hersteller
    ]
    
    print("🤖 Intelligente Techniker-Suche Demo\n")
    
    for query in test_queries:
        print(f"{'='*50}")
        result = engine.smart_search(query)
        
        print(f"🔍 Query: '{result['query']}'")
        print(f"🎯 Erkannt: {result['detected']}")
        print(f"📊 Ergebnisse: {len(result['results'])}")
        
        if result['suggestions']:
            for suggestion in result['suggestions']:
                print(f"   {suggestion}")
        
        print()

if __name__ == "__main__":
    demo_search()
