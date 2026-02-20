"""Resource monitoring utilities for tracking CPU and memory usage during computation.

Classes
-------
ResourceMonitor
    Monitor CPU and memory usage during multiprocessing computation.
"""

import os
import time
import threading
import logging

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

__all__ = ['ResourceMonitor', 'HAS_PSUTIL']


class ResourceMonitor:
    """Monitor CPU and memory usage during computation.
    
    Parameters
    ----------
    interval : float, optional
        Monitoring interval in seconds. Default is 5.0.
    log_to_file : str, optional
        Path to log file for recording history. Default is None.
    
    Attributes
    ----------
    history : list
        List of resource usage snapshots.
    
    Examples
    --------
    >>> monitor = ResourceMonitor(interval=2.0)
    >>> monitor.start()
    >>> # ... run computation ...
    >>> history = monitor.stop()
    >>> monitor.plot('resource_usage.png')
    """
    
    def __init__(self, interval=5.0, log_to_file=None):
        if not HAS_PSUTIL:
            raise ImportError(
                "psutil is required for ResourceMonitor. "
                "Install it with: pip install psutil"
            )
        
        self.interval = interval
        self.log_to_file = log_to_file
        self._running = False
        self._thread = None
        self.history = []
        self.logger = logging.getLogger('ResourceMonitor')
        self._start_time = None
        # Track CPU times per PID for proper percentage calculation
        self._cpu_times_cache = {}  # pid -> (timestamp, cpu_seconds)
        
    def start(self):
        """Start monitoring in background thread."""
        if self._running:
            self.logger.warning("Monitor already running")
            return
            
        self._running = True
        self._start_time = time.time()
        self.history = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"Resource monitoring started (interval={self.interval}s)")
        
    def stop(self):
        """Stop monitoring and return collected history.
        
        Returns
        -------
        list
            List of resource usage snapshots.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval + 1.0)
        
        duration = time.time() - self._start_time if self._start_time else 0
        self.logger.info(f"Resource monitoring stopped after {duration:.1f}s ({len(self.history)} samples)")
        
        return self.history
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        process = psutil.Process(os.getpid())
        
        # Reset CPU times cache
        self._cpu_times_cache = {}
        
        # Initialize CPU tracking for main process
        self._init_cpu_tracking(process)
        
        while self._running:
            try:
                snapshot = self._take_snapshot(process)
                self.history.append(snapshot)
                
                self.logger.info(
                    f"[Monitor] Workers: {snapshot['n_workers']:2d} | "
                    f"CPU: {snapshot['total_cpu_percent']:6.1f}% | "
                    f"Mem: {snapshot['total_mem_gb']:6.2f} GB | "
                    f"System: {snapshot['system_mem_percent']:.1f}% used | "
                    f"Avail: {snapshot['system_mem_available_gb']:.1f} GB"
                )
                
                if self.log_to_file:
                    self._write_to_file(snapshot)
                    
            except Exception as e:
                self.logger.warning(f"Monitor error: {e}")
            
            time.sleep(self.interval)
    
    def _init_cpu_tracking(self, proc):
        """Initialize CPU time tracking for a process.
        
        Parameters
        ----------
        proc : psutil.Process
            The process to start tracking.
        """
        try:
            cpu_times = proc.cpu_times()
            total_cpu = cpu_times.user + cpu_times.system
            self._cpu_times_cache[proc.pid] = (time.time(), total_cpu)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    def _get_cpu_percent(self, proc):
        """Calculate CPU percent for a process using cpu_times delta.
        
        This method properly handles newly spawned processes by tracking
        CPU time deltas rather than relying on psutil's cpu_percent()
        which returns 0 on first call.
        
        Parameters
        ----------
        proc : psutil.Process
            The process to get CPU percent for.
            
        Returns
        -------
        float
            CPU percent (can exceed 100% on multi-core systems).
        """
        try:
            cpu_times = proc.cpu_times()
            total_cpu = cpu_times.user + cpu_times.system
            now = time.time()
            
            if proc.pid in self._cpu_times_cache:
                prev_time, prev_cpu = self._cpu_times_cache[proc.pid]
                elapsed = now - prev_time
                if elapsed > 0:
                    cpu_percent = ((total_cpu - prev_cpu) / elapsed) * 100.0
                else:
                    cpu_percent = 0.0
            else:
                # First measurement for this process
                cpu_percent = 0.0
            
            # Update cache
            self._cpu_times_cache[proc.pid] = (now, total_cpu)
            return cpu_percent
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0

    def _take_snapshot(self, process):
        """Take a single snapshot of resource usage.
        
        Parameters
        ----------
        process : psutil.Process
            The main process to monitor.
            
        Returns
        -------
        dict
            Resource usage snapshot.
        """
        # Main process info
        mem_info = process.memory_info()
        main_cpu = self._get_cpu_percent(process)
        main_mem = mem_info.rss
        
        # Child processes (workers)
        # Filter out Python multiprocessing helper processes (semaphore/resource trackers)
        children = []
        for child in process.children(recursive=True):
            try:
                name = child.name()
                # Skip multiprocessing internal helper processes
                if 'semaphore_tracker' in name or 'resource_tracker' in name:
                    continue
                # Also check cmdline for these trackers (name might just be 'python')
                try:
                    cmdline = ' '.join(child.cmdline())
                    if 'semaphore_tracker' in cmdline or 'resource_tracker' in cmdline:
                        continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
                children.append({
                    'pid': child.pid,
                    'cpu_percent': self._get_cpu_percent(child),
                    'mem_bytes': child.memory_info().rss,
                    'name': name,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        n_workers = len(children)
        workers_cpu = sum(c['cpu_percent'] for c in children)
        workers_mem = sum(c['mem_bytes'] for c in children)
        
        # System-wide info
        system_mem = psutil.virtual_memory()
        
        return {
            'time': time.time(),
            'elapsed': time.time() - self._start_time,
            'main_cpu_percent': main_cpu,
            'main_mem_gb': main_mem / (1024**3),
            'n_workers': n_workers,
            'workers_cpu_percent': workers_cpu,
            'workers_mem_gb': workers_mem / (1024**3),
            'total_cpu_percent': main_cpu + workers_cpu,
            'total_mem_gb': (main_mem + workers_mem) / (1024**3),
            'system_mem_percent': system_mem.percent,
            'system_mem_available_gb': system_mem.available / (1024**3),
            'system_mem_total_gb': system_mem.total / (1024**3),
            'workers': children,
        }
    
    def _write_to_file(self, snapshot):
        """Write snapshot to log file."""
        with open(self.log_to_file, 'a') as f:
            f.write(
                f"{snapshot['elapsed']:.1f},"
                f"{snapshot['n_workers']},"
                f"{snapshot['total_cpu_percent']:.1f},"
                f"{snapshot['total_mem_gb']:.2f},"
                f"{snapshot['system_mem_percent']:.1f}\n"
            )
    
    def summary(self):
        """Print summary statistics of the monitoring session.
        
        Returns
        -------
        dict
            Summary statistics.
        """
        if not self.history:
            self.logger.warning("No history to summarize")
            return {}
        
        stats = {
            'duration_s': self.history[-1]['elapsed'],
            'n_samples': len(self.history),
            'max_workers': max(s['n_workers'] for s in self.history),
            'max_total_mem_gb': max(s['total_mem_gb'] for s in self.history),
            'avg_total_mem_gb': sum(s['total_mem_gb'] for s in self.history) / len(self.history),
            'max_cpu_percent': max(s['total_cpu_percent'] for s in self.history),
            'avg_cpu_percent': sum(s['total_cpu_percent'] for s in self.history) / len(self.history),
            'max_system_mem_percent': max(s['system_mem_percent'] for s in self.history),
        }
        
        self.logger.info("=" * 50)
        self.logger.info("Resource Monitor Summary")
        self.logger.info("=" * 50)
        self.logger.info(f"Duration: {stats['duration_s']:.1f}s")
        self.logger.info(f"Max workers: {stats['max_workers']}")
        self.logger.info(f"Max memory: {stats['max_total_mem_gb']:.2f} GB (avg: {stats['avg_total_mem_gb']:.2f} GB)")
        self.logger.info(f"Max CPU: {stats['max_cpu_percent']:.1f}% (avg: {stats['avg_cpu_percent']:.1f}%)")
        self.logger.info(f"Max system memory: {stats['max_system_mem_percent']:.1f}%")
        self.logger.info("=" * 50)
        
        return stats
    
    def plot(self, filename='resource_usage.png', show=False):
        """Plot resource usage history.
        
        Parameters
        ----------
        filename : str, optional
            Output filename for the plot. Default is 'resource_usage.png'.
        show : bool, optional
            Whether to display the plot. Default is False.
            
        Returns
        -------
        matplotlib.figure.Figure or None
            The figure object if matplotlib is available.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
            return None
        
        if not self.history:
            self.logger.warning("No history to plot")
            return None
        
        elapsed = [s['elapsed'] / 60 for s in self.history]  # Convert to minutes
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        # CPU usage
        ax = axes[0]
        ax.fill_between(elapsed, 0, [s['workers_cpu_percent'] for s in self.history], 
                       alpha=0.7, label='Workers', color='C0')
        ax.fill_between(elapsed, [s['workers_cpu_percent'] for s in self.history],
                       [s['total_cpu_percent'] for s in self.history],
                       alpha=0.7, label='Main', color='C1')
        ax.set_ylabel('CPU Usage (%)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Resource Usage Over Time')
        
        # Memory usage
        ax = axes[1]
        ax.fill_between(elapsed, 0, [s['workers_mem_gb'] for s in self.history],
                       alpha=0.7, label='Workers', color='C0')
        ax.fill_between(elapsed, [s['workers_mem_gb'] for s in self.history],
                       [s['total_mem_gb'] for s in self.history],
                       alpha=0.7, label='Main', color='C1')
        ax.axhline(self.history[0]['system_mem_total_gb'], color='r', 
                  linestyle='--', alpha=0.5, label='Total System')
        ax.set_ylabel('Memory (GB)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Worker count
        ax = axes[2]
        ax.plot(elapsed, [s['n_workers'] for s in self.history], 
               'g-', linewidth=2, marker='o', markersize=3)
        ax.set_ylabel('Active Workers')
        ax.set_xlabel('Time (minutes)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved resource plot to {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def to_dataframe(self):
        """Convert history to pandas DataFrame.
        
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with resource history if pandas is available.
        """
        try:
            import pandas as pd
        except ImportError:
            self.logger.warning("pandas not available")
            return None
        
        if not self.history:
            return pd.DataFrame()
        
        # Extract scalar values (not the nested 'workers' list)
        records = [{k: v for k, v in s.items() if k != 'workers'} for s in self.history]
        return pd.DataFrame(records)
