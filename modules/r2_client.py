import os
import boto3
import hashlib
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class R2Client:
    """R2 Client für Cloud-Speicher (Cloudflare R2 oder S3-kompatibel)"""
    
    def __init__(self, config: Dict[str, Any] = None, supabase_client = None):
        """
        Initialisiert den R2-Client
        
        Args:
            config: Konfiguration für den R2-Client (optional)
            supabase_client: Supabase-Client für Datenbankzugriff (optional)
        """
        # Lade Credentials aus Umgebungsvariablen
        self.account_id = os.environ.get("R2_ACCOUNT_ID", "")
        self.access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
        self.secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")
        self.bucket_name = os.environ.get("R2_BUCKET_NAME", "")
        self.region = os.environ.get("R2_REGION", "auto")
        self.public_url = os.environ.get("R2_PUBLIC_URL", "")
        
        logger.info(f"R2 Public URL aus Umgebungsvariablen: {self.public_url}")
        
        # Dictionary für bereits hochgeladene Dateien und Image-Hashes
        self.uploaded_file_hashes: Dict[str, str] = {}  # Hash -> object_name
        self.image_hashes: Dict[str, str] = {}  # Hash -> URL
        self.supabase = supabase_client
        
        # Überschreibe mit Config-Werten, falls vorhanden
        if config and "r2" in config:
            r2_config = config.get("r2", {})
            self.account_id = r2_config.get("account_id", self.account_id)
            self.bucket_name = r2_config.get("bucket_name", self.bucket_name)
            self.region = r2_config.get("region", self.region)
            
            # Überschreibe Public URL mit dem Wert aus der Konfiguration, falls vorhanden
            if "public_url" in r2_config:
                self.public_url = r2_config.get("public_url")
                logger.info(f"R2 Public URL aus Konfiguration überschrieben: {self.public_url}")
            
        # Lade vorhandene Bild-Hashes aus der Datenbank, wenn Supabase-Client vorhanden
        if self.supabase:
            self.load_image_hashes_from_db()
        
        self.client = None
        
        # Initialisiere S3-Client (verwendet für R2)
        if self.access_key and self.secret_key and self.account_id:
            try:
                endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"
                self.client = boto3.client(
                    service_name='s3',
                    endpoint_url=endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name=self.region
                )
                logger.info("R2 Client initialisiert")
            except Exception as e:
                logger.error(f"Fehler beim Initialisieren des R2 Clients: {e}")
                self.client = None
        else:
            logger.warning("R2 Credentials nicht vollständig - Client nicht initialisiert")
    
    def is_connected(self) -> bool:
        """
        Prüft, ob der R2-Client verbunden ist
        
        Returns:
            bool: True, wenn verbunden, sonst False
        """
        return self.client is not None
    
    def upload_file(self, file_path: str, object_name: Optional[str] = None) -> str:
        """
        Lädt eine Datei in den R2-Bucket hoch
        
        Args:
            file_path: Pfad zur lokalen Datei
            object_name: Name des Objekts im Bucket (optional, Standard: Dateiname)
            
        Returns:
            str: URL der hochgeladenen Datei oder leerer String bei Fehler
        """
        if not self.is_connected():
            logger.error("R2 Client nicht verbunden")
            return ""
            
        # Wenn kein Objektname angegeben, verwende Dateinamen
        if object_name is None:
            object_name = os.path.basename(file_path)
        
        # Prüfe, ob dieselbe Datei bereits hochgeladen wurde
        file_hash = self.get_file_hash(file_path)
        existing_object = self.uploaded_file_hashes.get(file_hash)
        
        if existing_object:
            logger.info(f"Datei mit Hash {file_hash} wurde bereits als {existing_object} hochgeladen")
            # URL für die existierende Datei zurückgeben
            if self.public_url:
                return f"{self.public_url.rstrip('/')}/{existing_object}"
            else:
                return f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{existing_object}"
            
        try:
            # Datei hochladen - plattformunabhängig mit Pfadnormalisierung
            norm_path = os.path.normpath(file_path)  # Normalisiert Pfade je nach Betriebssystem
            self.client.upload_file(norm_path, self.bucket_name, object_name)
            
            # Hash in Dictionary speichern
            self.uploaded_file_hashes[file_hash] = object_name
            
            # URL generieren
            if self.public_url:
                return f"{self.public_url.rstrip('/')}/{object_name}"
            else:
                # Fallback URL (funktioniert nur, wenn Bucket öffentlich ist)
                return f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{object_name}"
                
        except Exception as e:
            logger.error(f"Fehler beim Hochladen von {file_path}: {e}")
            return ""
    
    def upload_fileobj(self, file_obj, object_name: str, image_hash: str = None) -> str:
        """
        Lädt ein Dateiobjekt in den R2-Bucket hoch
        
        Args:
            file_obj: Dateiobjekt (z.B. BytesIO)
            object_name: Name des Objekts im Bucket
            image_hash: Hash des Bildes (optional)
            
        Returns:
            str: URL der hochgeladenen Datei oder leerer String bei Fehler
        """
        if not self.is_connected():
            logger.error("R2 Client nicht verbunden")
            return ""
        
        # Prüfe, ob ein Bild mit demselben Hash bereits existiert
        if image_hash and image_hash in self.image_hashes:
            logger.info(f"Bild mit Hash {image_hash} wurde bereits hochgeladen")
            return self.image_hashes[image_hash]
            
        try:
            # Datei hochladen - mit normalisiertem Objektnamen
            norm_object_name = object_name.replace('\\', '/')  # Stellt sicher, dass Objektnamen immer / verwenden
            self.client.upload_fileobj(file_obj, self.bucket_name, norm_object_name)
            
            # URL generieren
            url = ""
            if self.public_url:
                url = f"{self.public_url.rstrip('/')}/{object_name}"
            else:
                # Fallback URL
                url = f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{object_name}"
            
            # Hash speichern, wenn vorhanden
            if image_hash:
                self.image_hashes[image_hash] = url
                
            return url
                
        except Exception as e:
            logger.error(f"Fehler beim Hochladen von Dateiobjekt: {e}")
            return ""
            
    def get_file_hash(self, file_path: str) -> str:
        """
        Berechnet den SHA-256 Hash einer Datei
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            str: SHA-256 Hash der Datei oder leerer String bei Fehler
        """
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Fehler beim Berechnen des Hashes für {file_path}: {e}")
            return ""
            
    def hash_exists(self, image_hash: str) -> bool:
        """
        Prüft, ob ein Bild mit dem angegebenen Hash bereits hochgeladen wurde
        
        Args:
            image_hash: Hash des Bildes
            
        Returns:
            bool: True, wenn bereits hochgeladen, sonst False
        """
        # Zuerst im lokalen Cache prüfen
        if image_hash in self.image_hashes:
            return True
            
        # Falls nicht im Cache, in der Datenbank suchen
        if hasattr(self, "check_hash_in_db"):
            url = self.check_hash_in_db(image_hash)
            return bool(url)  # True wenn eine URL gefunden wurde, False sonst
            
        return False
        
    def get_url_for_hash(self, image_hash: str) -> str:
        """
        Gibt die URL für ein bereits hochgeladenes Bild zurück
        
        Args:
            image_hash: Hash des Bildes
            
        Returns:
            str: URL des Bildes oder leerer String, wenn nicht gefunden
        """
        # Zuerst im lokalen Cache prüfen
        url = self.image_hashes.get(image_hash, "")
        if url:
            return url
            
        # Falls nicht im Cache, in der Datenbank suchen
        if hasattr(self, "supabase") and self.supabase:
            db_url = self.check_hash_in_db(image_hash)
            if db_url:
                # Speichere im lokalen Cache für zukünftige Abfragen
                self.image_hashes[image_hash] = db_url
                return db_url
            
        return ""
        
    def load_image_hashes_from_db(self):
        """
        Lädt alle vorhandenen Bild-Hashes aus der Datenbank
        """
        if not self.supabase:
            logger.warning("Kein Supabase-Client vorhanden, kann Bild-Hashes nicht laden")
            return
            
        try:
            # Alle Einträge mit nicht-leeren hash-Werten holen
            response = self.supabase.table("images").select("hash, storage_url").not_("hash", "is", "").execute()
            
            if response.data:
                logger.info(f"Lade {len(response.data)} Bild-Hashes aus der Datenbank")
                for item in response.data:
                    image_hash = item.get("hash")
                    image_url = item.get("storage_url")
                    if image_hash and image_url:
                        self.image_hashes[image_hash] = image_url
                        
                logger.info(f"Erfolgreich {len(self.image_hashes)} Bild-Hashes geladen")
            else:
                logger.info("Keine Bild-Hashes in der Datenbank gefunden")
                
        except Exception as e:
            logger.error(f"Fehler beim Laden der Bild-Hashes aus der Datenbank: {e}")
            
    def check_hash_in_db(self, image_hash: str) -> str:
        """
        Prüft, ob ein Bild-Hash in der Datenbank existiert
        
        Args:
            image_hash: Hash des Bildes
            
        Returns:
            str: URL des Bildes oder leerer String, wenn nicht gefunden
        """
        if not self.supabase or not image_hash:
            return ""
            
        # In der Datenbank suchen
        try:
            response = self.supabase.table("images").select("storage_url").eq("hash", image_hash).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                image_url = response.data[0].get("storage_url", "")
                return image_url
                    
            return ""
            
        except Exception as e:
            logger.error(f"Fehler beim Prüfen des Hashes in der Datenbank: {e}")
            return ""