from pathlib import Path
from typing import Set


class DoneTracker:
    """
    Handles tracking of completed/done items using a text file.
    
    This class provides a simple interface for tracking which items
    (typically image names) have been processed/completed.
    """
    
    def __init__(self, output_dir: str | Path, filename: str = "done_names.txt"):
        """
        Initialize the done tracker.
        
        Args:
            output_dir: Directory where the done file will be stored
            filename: Name of the file to track done items (default: "done_names.txt")
        """
        self._output_dir = Path(output_dir)
        self._filename = filename
        self._done: Set[str] = self._load_done()
    
    @property
    def _done_file(self) -> Path:
        """Get the path to the done tracking file."""
        return self._output_dir / self._filename
    
    def _load_done(self) -> Set[str]:
        """Load the set of done items from the file."""
        if self._done_file.exists():
            return set(self._done_file.read_text().splitlines())
        return set()
    
    def mark_done(self, name: str) -> None:
        """
        Mark an item as done.
        
        Args:
            name: The name/identifier of the item to mark as done
        """
        self._done.add(name)
        with self._done_file.open("a") as f:
            f.write(name + "\n")
    
    def is_done(self, name: str) -> bool:
        """
        Check if an item is marked as done.
        
        Args:
            name: The name/identifier to check
            
        Returns:
            True if the item is marked as done, False otherwise
        """
        return name in self._done
    
    def get_done_count(self) -> int:
        """Get the number of items marked as done."""
        return len(self._done)
    
    def get_done_items(self) -> Set[str]:
        """Get a copy of all done items."""
        return self._done.copy()
    
    def clear_done(self) -> None:
        """Clear all done items (removes the file)."""
        if self._done_file.exists():
            self._done_file.unlink()
        self._done.clear()


class SkippedTracker(DoneTracker):
    """
    Handles tracking of skipped items using a text file.
    
    Similar to DoneTracker but for items that were skipped/ignored.
    """
    
    def __init__(self, output_dir: str | Path, filename: str = "skipped_images.txt"):
        """
        Initialize the skipped tracker.
        
        Args:
            output_dir: Directory where the skipped file will be stored  
            filename: Name of the file to track skipped items (default: "skipped_images.txt")
        """
        super().__init__(output_dir, filename)
    
    def mark_skipped(self, name: str) -> None:
        """
        Mark an item as skipped.
        
        Args:
            name: The name/identifier of the item to mark as skipped
        """
        self.mark_done(name)  # Reuse the done logic
    
    def is_skipped(self, name: str) -> bool:
        """
        Check if an item is marked as skipped.
        
        Args:
            name: The name/identifier to check
            
        Returns:
            True if the item is marked as skipped, False otherwise
        """
        return self.is_done(name)  # Reuse the done logic
    
    def get_skipped_count(self) -> int:
        """Get the number of items marked as skipped."""
        return self.get_done_count()
    
    def get_skipped_items(self) -> Set[str]:
        """Get a copy of all skipped items."""
        return self.get_done_items()