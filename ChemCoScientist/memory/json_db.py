import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

class JSONFileDB:
    def __init__(self, db_path: str = "file_db.json"):
        self.db_path = Path(db_path)
        self.db = self._load_db()
    
    def _load_db(self) -> Dict:
        """Load database from JSON file"""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self._create_empty_db()
    
    def _create_empty_db(self) -> Dict:
        """Create empty database structure"""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "files": {},
            "sessions": {},
            "indices": {
                "by_filename": {},
                "by_tags": {},
                "by_user": {},
                "by_date": {}
            }
        }
    
    def _save_db(self):
        """Save database to JSON file"""
        self.db["last_updated"] = datetime.now().isoformat()
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.db, f, indent=2, ensure_ascii=False)
    
    def add_file(self, file_path: str, original_filename: str, file_size: int, 
                 uploaded_by: str, user_context: str = None, **kwargs) -> str:
        """Add a new file to the database"""

        # Check if file with same name already exists
        filename = Path(file_path).name
        existing_file_id = self.db["indices"]["by_filename"].get(filename)
        
        if existing_file_id and existing_file_id in self.db["files"]:
            # Update existing file instead of creating new one
            return self._update_existing_file(existing_file_id, file_path, original_filename, 
                                            file_size, uploaded_by, user_context, **kwargs)
        
        # Create new file entry
        file_id = f"file_{uuid.uuid4().hex[:8]}"
        
        # Generate CSV metadata if it's a CSV file
        csv_metadata = None
        if file_path.endswith('.csv'):
            csv_metadata = self._generate_csv_metadata(file_path)
        
        file_data = {
            "id": file_id,
            "filename": Path(file_path).name,
            "original_filename": original_filename,
            "file_path": file_path,
            "file_size": file_size,
            "file_type": Path(file_path).suffix[1:],
            "mime_type": self._get_mime_type(file_path),
            
            "upload_date": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "upload_source": kwargs.get('upload_source', 'user_upload'),
            
            "uploaded_by": uploaded_by,
            "user_context": user_context,
            
            "processing_status": "completed",
            "processing_error": None,
            
            "metadata": {
                "csv_metadata": csv_metadata
            } if csv_metadata else {},
            
            "tags": self._generate_auto_tags(file_path, user_context, csv_metadata),
            "access_logs": []
        }
        
        # Add to database
        self.db["files"][file_id] = file_data
        self._update_indices(file_id, file_data)
        self._save_db()
        
        return file_id

    def _update_tags(self, existing_tags: List[Dict], file_path: str, 
                    user_context: str, csv_metadata: Dict = None) -> List[Dict]:
        """Update tags while preserving manual tags and updating auto-generated ones"""
        # Separate manual and auto-generated tags
        manual_tags = [tag for tag in existing_tags if tag.get("type") != "auto_generated"]
        auto_tags = self._generate_auto_tags(file_path, user_context, csv_metadata)
        
        return manual_tags + auto_tags
        
    def _update_existing_file(self, file_id: str, file_path: str, original_filename: str, 
                            file_size: int, uploaded_by: str, user_context: str = None, 
                            **kwargs) -> str:
        """Update an existing file with new data and metadata"""
        existing_file = self.db["files"][file_id]
        
        # Store old file path for potential cleanup
        old_file_path = existing_file["file_path"]
        
        # Generate new CSV metadata if it's a CSV file
        csv_metadata = None
        if file_path.endswith('.csv'):
            csv_metadata = self._generate_csv_metadata(file_path)
        
        # Update file data while preserving some existing metadata
        updated_file_data = {
            "id": file_id,
            "filename": Path(file_path).name,
            "original_filename": original_filename,
            "file_path": file_path,
            "file_size": file_size,
            "file_type": Path(file_path).suffix[1:],
            "mime_type": self._get_mime_type(file_path),
            
            # Update timestamps
            "upload_date": existing_file["upload_date"],  # Keep original upload date
            "last_updated": datetime.now().isoformat(),   # Add update timestamp
            "last_accessed": datetime.now().isoformat(),
            "upload_source": kwargs.get('upload_source', existing_file.get('upload_source', 'user_upload')),
            
            # Update user info
            "uploaded_by": uploaded_by,
            "user_context": user_context or existing_file.get('user_context'),
            
            "processing_status": "completed",
            "processing_error": None,
            
            # Update metadata
            "metadata": {
                "csv_metadata": csv_metadata
            } if csv_metadata else {},
            
            # Generate new tags but preserve manual tags if any
            "tags": self._update_tags(existing_file.get("tags", []), file_path, user_context, csv_metadata),
            
            # Preserve access logs history
            "access_logs": existing_file.get("access_logs", [])
        }
        
        # Remove old file from indices before updating
        self._remove_from_indices(file_id, existing_file)
        
        # Update the file entry
        self.db["files"][file_id] = updated_file_data
        
        # Update indices with new data
        self._update_indices(file_id, updated_file_data)
        
        # Optionally delete the old physical file if it's different from the new one
        if old_file_path != file_path and os.path.exists(old_file_path):
            try:
                os.remove(old_file_path)
                print(f"Deleted old physical file: {old_file_path}")
            except OSError as e:
                print(f"Warning: Could not delete old physical file {old_file_path}: {e}")
        
        self._save_db()
        return file_id

    def _generate_csv_metadata(self, file_path: str) -> Optional[Dict]:
        """Generate CSV metadata using pandas"""
        try:
            df = pd.read_csv(file_path)
            
            # Sample data (first 2 rows)
            sample_data = df.head(2).to_dict('records')
            
            # Infer column types
            column_types = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    column_types[col] = "numeric"
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    column_types[col] = "datetime"
                else:
                    column_types[col] = "string"
            
            # Basic data summary for numeric columns
            data_summary = {}
            for col in df.select_dtypes(include=['number']).columns:
                data_summary[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "avg": float(df[col].mean()),
                    "total": float(df[col].sum())
                }
            
            return {
                "column_count": len(df.columns),
                "row_count": len(df),
                "headers": df.columns.tolist(),
                "column_types": column_types,
                "sample_data": sample_data,
                "has_headers": True,
                "delimiter": ",",
                "encoding": "utf-8",
                "data_summary": data_summary
            }
        except Exception as e:
            print(f"Error generating CSV metadata: {e}")
            return None
    
    def _generate_auto_tags(self, file_path: str, user_context: str, 
                           csv_metadata: Dict = None) -> List[Dict]:
        """Generate automatic tags based on filename and content"""
        #TODO
        return []
        # tags = []
        # filename = Path(file_path).name.lower()
        
        # # Filename-based tags
        # if 'sales' in filename:
        #     tags.append({"name": "sales", "type": "auto_generated", "confidence": 0.9, "created_by": "system"})
        # if 'customer' in filename or 'feedback' in filename:
        #     tags.append({"name": "feedback", "type": "auto_generated", "confidence": 0.9, "created_by": "system"})
        
        # # Context-based tags
        # if user_context:
        #     context_lower = user_context.lower()
        #     if 'analyze' in context_lower:
        #         tags.append({"name": "analysis", "type": "auto_generated", "confidence": 0.8, "created_by": "system"})
        #     if 'financial' in context_lower:
        #         tags.append({"name": "financial", "type": "auto_generated", "confidence": 0.8, "created_by": "system"})
        
        # # CSV content-based tags
        # if csv_metadata and 'headers' in csv_metadata:
        #     headers = [h.lower() for h in csv_metadata['headers']]
        #     if any('profit' in h for h in headers):
        #         tags.append({"name": "profit", "type": "auto_generated", "confidence": 0.7, "created_by": "system"})
        #     if any('rating' in h for h in headers):
        #         tags.append({"name": "ratings", "type": "auto_generated", "confidence": 0.7, "created_by": "system"})
        
        # return tags
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type based on file extension"""
        extensions = {
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        ext = Path(file_path).suffix.lower()
        return extensions.get(ext, 'application/octet-stream')
    
    def _update_indices(self, file_id: str, file_data: Dict):
        """Update all indices for a file"""
        # By filename
        self.db["indices"]["by_filename"][file_data["filename"]] = file_id
        
        # By user
        user = file_data["uploaded_by"]
        if user not in self.db["indices"]["by_user"]:
            self.db["indices"]["by_user"][user] = []
        if file_id not in self.db["indices"]["by_user"][user]:
            self.db["indices"]["by_user"][user].append(file_id)
        
        # By date
        upload_date = file_data["upload_date"][:10]  # YYYY-MM-DD
        if upload_date not in self.db["indices"]["by_date"]:
            self.db["indices"]["by_date"][upload_date] = []
        if file_id not in self.db["indices"]["by_date"][upload_date]:
            self.db["indices"]["by_date"][upload_date].append(file_id)
        
        # By tags
        for tag in file_data.get("tags", []):
            tag_name = tag["name"]
            if tag_name not in self.db["indices"]["by_tags"]:
                self.db["indices"]["by_tags"][tag_name] = []
            if file_id not in self.db["indices"]["by_tags"][tag_name]:
                self.db["indices"]["by_tags"][tag_name].append(file_id)
    
    def log_file_access(self, file_id: str, accessed_by: str, access_type: str, 
                       user_query: str = None, agent_actions: List[str] = None,
                       session_id: str = None) -> bool:
        """Log file access for memory mechanism"""
        if file_id not in self.db["files"]:
            return False
        
        access_log = {
            "id": f"log_{uuid.uuid4().hex[:6]}",
            "accessed_by": accessed_by,
            "access_type": access_type,
            "access_timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "agent_actions": agent_actions or [],
            "accessed_columns": self._extract_accessed_columns(user_query),
            "success": True,
            "session_id": session_id
        }
        
        self.db["files"][file_id]["access_logs"].append(access_log)
        self.db["files"][file_id]["last_accessed"] = datetime.now().isoformat()
        
        # Update session if provided
        if session_id:
            if session_id not in self.db["sessions"]:
                self.db["sessions"][session_id] = {
                    "session_id": session_id,
                    "user_id": accessed_by,
                    "start_time": datetime.now().isoformat(),
                    "last_activity": datetime.now().isoformat(),
                    "files_accessed": [file_id],
                    "conversation_context": user_query
                }
            else:
                session = self.db["sessions"][session_id]
                session["last_activity"] = datetime.now().isoformat()
                if file_id not in session["files_accessed"]:
                    session["files_accessed"].append(file_id)
        
        self._save_db()
        return True
    
    def _extract_accessed_columns(self, user_query: str) -> List[str]:
        """Extract column names from user query (simple implementation)"""
        #TODO
        return []
        # if not user_query:
        #     return []
        
        # # This is a simple implementation - you might want to use NLP for better extraction
        # common_columns = ['date', 'sales', 'profit', 'revenue', 'customer', 'product', 'region', 'rating']
        # accessed = []
        
        # query_lower = user_query.lower()
        # for col in common_columns:
        #     if col in query_lower:
        #         accessed.append(col)
        
        # return accessed
    
    # Query methods
    def get_file_by_id(self, file_id: str) -> Optional[Dict]:
        return self.db["files"].get(file_id)
    
    def get_file_by_filename(self, filename: str) -> Optional[Dict]:
        file_id = self.db["indices"]["by_filename"].get(filename)
        return self.get_file_by_id(file_id) if file_id else None
    
    def get_files_by_user(self, user_id: str) -> List[Dict]:
        file_ids = self.db["indices"]["by_user"].get(user_id, [])
        return [self.db["files"][fid] for fid in file_ids if fid in self.db["files"]]
    
    def get_files_by_tag(self, tag: str) -> List[Dict]:
        file_ids = self.db["indices"]["by_tags"].get(tag, [])
        return [self.db["files"][fid] for fid in file_ids if fid in self.db["files"]]
    
    def search_files(self, query: str) -> List[Dict]:
        """Simple search across filenames and user context"""
        results = []
        query_lower = query.lower()
        
        for file_id, file_data in self.db["files"].items():
            if (query_lower in file_data["filename"].lower() or 
                query_lower in file_data.get("user_context", "").lower() or
                any(query_lower in tag["name"].lower() for tag in file_data.get("tags", []))):
                results.append(file_data)
        
        return results
    
    def get_recent_files(self, limit: int = 10) -> List[Dict]:
        """Get most recently accessed files"""
        files = list(self.db["files"].values())
        files.sort(key=lambda x: x["last_accessed"], reverse=True)
        return files[:limit]


    def remove_file(self, file_id: str, delete_physical_file: bool = True) -> bool:
        """Remove a file from the database and optionally delete the physical file.
        
        Args:
            file_id: The ID of the file to remove
            delete_physical_file: If True, also delete the physical file from disk
            
        Returns:
            bool: True if successful, False if file not found
        """
        if file_id not in self.db["files"]:
            return False
        
        # Get file data before removal (for physical file deletion)
        file_data = self.db["files"][file_id]
        file_path = file_data["file_path"]
        
        # Remove from files dictionary
        del self.db["files"][file_id]
        
        # Remove from all indices
        self._remove_from_indices(file_id, file_data)
        
        # Remove from sessions
        self._remove_file_from_sessions(file_id)
        
        # Delete physical file if requested
        if delete_physical_file and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted physical file: {file_path}")
            except OSError as e:
                print(f"Warning: Could not delete physical file {file_path}: {e}")
        
        self._save_db()
        return True
    
    def remove_file_by_filename(self, filename: str, delete_physical_file: bool = True) -> bool:
        """Remove a file by filename.
        
        Args:
            filename: The filename to remove
            delete_physical_file: If True, also delete the physical file from disk
            
        Returns:
            bool: True if successful, False if file not found
        """
        file_id = self.db["indices"]["by_filename"].get(filename)
        if file_id:
            return self.remove_file(file_id, delete_physical_file)
        return False
    
    def remove_files_by_user(self, user_id: str, delete_physical_files: bool = True) -> int:
        """Remove all files uploaded by a specific user.
        
        Args:
            user_id: The user ID whose files should be removed
            delete_physical_files: If True, also delete the physical files from disk
            
        Returns:
            int: Number of files removed
        """
        file_ids = self.db["indices"]["by_user"].get(user_id, [])[:]  # Copy the list
        removed_count = 0
        
        for file_id in file_ids:
            if self.remove_file(file_id, delete_physical_files):
                removed_count += 1
        
        return removed_count
    
    def remove_files_by_tag(self, tag: str, delete_physical_files: bool = True) -> int:
        """Remove all files with a specific tag.
        
        Args:
            tag: The tag to match
            delete_physical_files: If True, also delete the physical files from disk
            
        Returns:
            int: Number of files removed
        """
        file_ids = self.db["indices"]["by_tags"].get(tag, [])[:]  # Copy the list
        removed_count = 0
        
        for file_id in file_ids:
            if self.remove_file(file_id, delete_physical_files):
                removed_count += 1
        
        return removed_count
    
    def cleanup_orphaned_files(self, uploads_directory: str = "data_store/datasets") -> int:
        """Remove database entries for files that no longer exist physically.
        
        Args:
            uploads_directory: Directory to check for physical files
            
        Returns:
            int: Number of orphaned entries removed
        """
        removed_count = 0
        file_ids_to_remove = []
        
        for file_id, file_data in self.db["files"].items():
            file_path = file_data["file_path"]
            if not os.path.exists(file_path):
                file_ids_to_remove.append(file_id)
        
        for file_id in file_ids_to_remove:
            if self.remove_file(file_id, delete_physical_file=False):
                removed_count += 1
                print(f"Removed orphaned database entry: {file_id}")
        
        return removed_count
    
    def _remove_from_indices(self, file_id: str, file_data: Dict):
        """Remove a file from all indices."""
        # Remove from filename index
        filename = file_data["filename"]
        if filename in self.db["indices"]["by_filename"]:
            if self.db["indices"]["by_filename"][filename] == file_id:
                del self.db["indices"]["by_filename"][filename]
        
        # Remove from user index
        user_id = file_data["uploaded_by"]
        if user_id in self.db["indices"]["by_user"]:
            if file_id in self.db["indices"]["by_user"][user_id]:
                self.db["indices"]["by_user"][user_id].remove(file_id)
                # Remove empty user arrays
                if not self.db["indices"]["by_user"][user_id]:
                    del self.db["indices"]["by_user"][user_id]
        
        # Remove from date index
        upload_date = file_data["upload_date"][:10]
        if upload_date in self.db["indices"]["by_date"]:
            if file_id in self.db["indices"]["by_date"][upload_date]:
                self.db["indices"]["by_date"][upload_date].remove(file_id)
                # Remove empty date arrays
                if not self.db["indices"]["by_date"][upload_date]:
                    del self.db["indices"]["by_date"][upload_date]
        
        # Remove from tag indices
        for tag in file_data.get("tags", []):
            tag_name = tag["name"]
            if tag_name in self.db["indices"]["by_tags"]:
                if file_id in self.db["indices"]["by_tags"][tag_name]:
                    self.db["indices"]["by_tags"][tag_name].remove(file_id)
                    # Remove empty tag arrays
                    if not self.db["indices"]["by_tags"][tag_name]:
                        del self.db["indices"]["by_tags"][tag_name]
    
    def _remove_file_from_sessions(self, file_id: str):
        """Remove file reference from all sessions."""
        sessions_to_remove = []
        
        for session_id, session_data in self.db["sessions"].items():
            if file_id in session_data.get("files_accessed", []):
                session_data["files_accessed"].remove(file_id)
                # Mark empty sessions for removal
                if not session_data["files_accessed"]:
                    sessions_to_remove.append(session_id)
        
        # Remove empty sessions
        for session_id in sessions_to_remove:
            del self.db["sessions"][session_id]
    
    def get_file_stats(self) -> Dict[str, Any]:
        """Get statistics about the files in the database."""
        total_files = len(self.db["files"])
        total_size = sum(file_data["file_size"] for file_data in self.db["files"].values())
        
        file_types = {}
        for file_data in self.db["files"].values():
            file_type = file_data["file_type"]
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        users = {}
        for file_data in self.db["files"].values():
            user = file_data["uploaded_by"]
            users[user] = users.get(user, 0) + 1
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": file_types,
            "users": users,
            "sessions_count": len(self.db["sessions"])
        }