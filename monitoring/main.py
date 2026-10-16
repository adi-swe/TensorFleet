from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY
import os
import logging
import random
import time
import subprocess
import threading

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # In production, replace with specific origins like ["http://localhost:3000", "http://localhost:3002"]
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-User-ID", "x-user-id"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
job_submissions = Counter('tensorfleet_job_submissions_total', 'Total number of job submissions')
active_jobs = Gauge('tensorfleet_active_jobs', 'Number of currently active jobs')
active_workers = Gauge('tensorfleet_active_workers', 'Number of active worker nodes')
task_duration = Histogram('tensorfleet_task_duration_seconds', 'Task execution duration')
training_loss = Gauge('tensorfleet_training_loss', 'Current training loss', ['job_id'])
training_accuracy = Gauge('tensorfleet_training_accuracy', 'Current training accuracy', ['job_id'])
desired_workers = Gauge('tensorfleet_desired_workers', 'Desired number of worker nodes')

# Mock data for demonstration
jobs_data = {}
workers_data = {}
scaling_config = {
    'current_workers': 3,
    'desired_workers': 3,
    'min_workers': 1,
    'max_workers': 10,
    'auto_scale_enabled': True,
    'scale_down_threshold': 0.3,  # Scale down if utilization < 30%
    'scale_up_threshold': 0.8,    # Scale up if utilization > 80%
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'monitoring'}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}

@app.route('/api/v1/metrics/jobs', methods=['GET'])
def get_job_metrics():
    """Get aggregated job metrics"""
    total_jobs = len(jobs_data)
    running_jobs = sum(1 for j in jobs_data.values() if j.get('status') == 'RUNNING')
    completed_jobs = sum(1 for j in jobs_data.values() if j.get('status') == 'COMPLETED')
    failed_jobs = sum(1 for j in jobs_data.values() if j.get('status') == 'FAILED')

    return jsonify({
        'total_jobs': total_jobs,
        'running_jobs': running_jobs,
        'completed_jobs': completed_jobs,
        'failed_jobs': failed_jobs,
        'active_workers': len(workers_data)
    }), 200

@app.route('/api/v1/metrics/jobs/<job_id>', methods=['GET'])
def get_job_metrics_detail(job_id):
    """Get detailed metrics for a specific job"""
    if job_id not in jobs_data:
        return jsonify({'error': 'Job not found', 'job_id': job_id}), 404

    job = jobs_data[job_id]
    return jsonify(job), 200

@app.route('/api/v1/metrics/jobs/<job_id>', methods=['POST'])
def update_job_metrics(job_id):
    """Update metrics for a specific job"""
    
    data = request.get_json()
    
    if job_id not in jobs_data:
        jobs_data[job_id] = {
            'job_id': job_id,
            'status': 'RUNNING',
            'start_time': time.time()
        }
        job_submissions.inc()
        active_jobs.inc()

    job = jobs_data[job_id]
    
    if 'status' in data:
        old_status = job.get('status')
        new_status = data['status']
        job['status'] = new_status
        
        if old_status == 'RUNNING' and new_status in ['COMPLETED', 'FAILED']:
            active_jobs.dec()
    
    if 'loss' in data:
        job['loss'] = data['loss']
        training_loss.labels(job_id=job_id).set(data['loss'])
    
    if 'accuracy' in data:
        job['accuracy'] = data['accuracy']
        training_accuracy.labels(job_id=job_id).set(data['accuracy'])
    
    if 'progress' in data:
        job['progress'] = data['progress']
    
    job['updated_at'] = time.time()

    return jsonify({'message': 'Metrics updated', 'job': job}), 200

@app.route('/api/v1/metrics/workers', methods=['GET'])
def get_worker_metrics():
    """Get metrics for all workers"""
    return jsonify({
        'workers': list(workers_data.values()),
        'total': len(workers_data)
    }), 200

@app.route('/api/v1/metrics/workers/<worker_id>', methods=['POST'])
def update_worker_metrics(worker_id):
    """Update metrics for a specific worker"""
    data = request.get_json()
    
    if worker_id not in workers_data:
        workers_data[worker_id] = {
            'worker_id': worker_id,
            'status': 'ACTIVE',
            'registered_at': time.time()
        }
        active_workers.inc()

    worker = workers_data[worker_id]
    worker.update(data)
    worker['updated_at'] = time.time()

    return jsonify({'message': 'Worker metrics updated', 'worker': worker}), 200

@app.route('/api/v1/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard summary data"""
    # Calculate system-wide metrics
    total_jobs = len(jobs_data)
    active_count = sum(1 for j in jobs_data.values() if j.get('status') == 'RUNNING')
    
    # Calculate average metrics
    avg_loss = 0
    avg_accuracy = 0
    if jobs_data:
        losses = [j.get('loss', 0) for j in jobs_data.values() if 'loss' in j]
        accuracies = [j.get('accuracy', 0) for j in jobs_data.values() if 'accuracy' in j]
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

    return jsonify({
        'system_status': 'HEALTHY',
        'total_jobs': total_jobs,
        'active_jobs': active_count,
        'active_workers': len(workers_data),
        'avg_loss': avg_loss,
        'avg_accuracy': avg_accuracy,
        'timestamp': time.time()
    }), 200

@app.route('/worker-activity', methods=['GET'])
def get_worker_activity():
    """Get real-time worker activity data for visualization"""
    current_time = time.time()
    
    # Format worker data for the WorkerVisualization component
    workers_list = []
    for worker_id, worker_data in workers_data.items():
        last_activity = worker_data.get('updated_at', worker_data.get('registered_at', current_time))
        
        # Determine worker status based on last activity (consider inactive if no update in 30 seconds)
        is_active = (current_time - last_activity) < 30
        status = worker_data.get('status', 'IDLE') if is_active else 'OFFLINE'
        
        workers_list.append({
            'worker_id': worker_id,
            'status': status,
            'current_task_id': worker_data.get('current_task_id', ''),
            'current_job_id': worker_data.get('current_job_id', ''),
            'tasks_completed': worker_data.get('tasks_completed', 0),
            'last_activity_time': last_activity,
            'cpu_usage': worker_data.get('cpu_usage'),
            'memory_usage': worker_data.get('memory_usage'),
            'uptime': current_time - worker_data.get('registered_at', current_time),
            'is_active': is_active
        })
    
    return jsonify({
        'workers': workers_list,
        'total_workers': len(workers_list),
        'active_workers': len([w for w in workers_list if w['is_active']]),
        'busy_workers': len([w for w in workers_list if w['status'] == 'BUSY']),
        'timestamp': current_time
    }), 200

@app.route('/api/v1/scaling/config', methods=['GET'])
def get_scaling_config():
    """Get current scaling configuration"""
    return jsonify(scaling_config), 200

@app.route('/api/v1/scaling/config', methods=['POST'])
def update_scaling_config():
    """Update scaling configuration"""
    data = request.get_json()
    
    # Update scaling configuration with new values
    for key, value in data.items():
        if key in scaling_config:
            scaling_config[key] = value
    
    return jsonify({'message': 'Scaling configuration updated', 'config': scaling_config}), 200

@app.route('/api/v1/scaling/workers', methods=['POST'])
def scale_workers():
    """Scale workers to a specific count"""
    data = request.get_json(silent=True) or {}
    target_count = data.get('worker_count', scaling_config['desired_workers'])
    return scale_workers_internal(target_count)

def scale_workers_internal(target_count):
    """Scale workers to target_count. Returns a (response, status) tuple."""
    # Validate worker count
    if target_count < scaling_config['min_workers']:
        return jsonify({'error': f'Worker count must be at least {scaling_config["min_workers"]}'}), 400
    
    if target_count > scaling_config['max_workers']:
        return jsonify({'error': f'Worker count cannot exceed {scaling_config["max_workers"]}'}), 400
    
    # Update desired workers
    old_count = scaling_config['desired_workers']
    scaling_config['desired_workers'] = target_count
    
    # Use docker-compose command to scale workers
    try:
        # Use subprocess to call docker compose scale command
        compose_file_path = '/app/docker-compose.yml'
        
        # Build the docker compose command
        cmd = [
            'docker', 'compose',
            '-f', compose_file_path,
            'up', '-d',
            '--scale', f'worker={target_count}',
            '--no-recreate'
        ]
        
        logger.info(f"Executing scaling command: {' '.join(cmd)}")
        
        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            logger.error(f"Failed to scale workers: {error_msg}")
            return jsonify({'error': f'Failed to scale workers: {error_msg}'}), 500
        
        scaling_config['current_workers'] = target_count
        desired_workers.set(target_count)
        active_workers.set(target_count)
        
        logger.info(f"✅ Successfully scaled workers from {old_count} to {target_count}")
        return jsonify({
            'message': f'Successfully scaled workers to {target_count}',
            'old_count': old_count,
            'new_count': target_count
        }), 200
    
    except Exception as e:
        logger.error(f"Error scaling workers: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/scaling/scale-up', methods=['POST'])
def scale_up_workers():
    """Scale up workers by 1"""
    target_count = min(scaling_config['desired_workers'] + 1, scaling_config['max_workers'])
    return scale_workers_internal(target_count)

@app.route('/api/v1/scaling/scale-down', methods=['POST'])
def scale_down_workers():
    """Scale down workers by 1"""
    target_count = max(scaling_config['desired_workers'] - 1, scaling_config['min_workers'])
    return scale_workers_internal(target_count)

# Guard so only ONE auto-shrink monitor thread ever runs, regardless of how
# many times the enable endpoint is called.
_auto_shrink_lock = threading.Lock()
_auto_shrink_thread = None

@app.route('/api/v1/scaling/auto-shrink', methods=['POST'])
def enable_auto_shrink():
    """Enable automatic shrinking of workers when jobs complete"""
    global _auto_shrink_thread
    data = request.get_json(silent=True) or {}
    enabled = data.get('enabled', True)

    scaling_config['auto_scale_enabled'] = enabled

    if enabled:
        # Start auto-shrink monitoring thread (at most one)
        with _auto_shrink_lock:
            if _auto_shrink_thread is None or not _auto_shrink_thread.is_alive():
                _auto_shrink_thread = threading.Thread(target=monitor_and_shrink, daemon=True)
                _auto_shrink_thread.start()
        return jsonify({'message': 'Auto-shrink enabled', 'config': scaling_config}), 200
    else:
        return jsonify({'message': 'Auto-shrink disabled', 'config': scaling_config}), 200

def monitor_and_shrink():
    """Monitor worker utilization and shrink when idle"""
    logger.info("🔄 Auto-shrink monitoring started")
    
    while scaling_config.get('auto_scale_enabled', False):
        try:
            # Calculate worker utilization
            total_workers = len(workers_data)
            busy_workers = len([w for w in workers_data.values() if w.get('status') == 'BUSY'])
            
            if total_workers > 0:
                utilization = busy_workers / total_workers
                
                # Scale down if utilization is low and we have more than min workers
                if utilization < scaling_config['scale_down_threshold'] and scaling_config['current_workers'] > scaling_config['min_workers']:
                    new_count = max(scaling_config['current_workers'] - 1, scaling_config['min_workers'])
                    logger.info(f"📉 Low utilization ({utilization:.1%}), scaling down to {new_count} workers")
                    
                    # Execute scale down
                    subprocess.run(
                        ['docker', 'compose', 'up', '-d', '--scale', f'worker={new_count}', '--no-recreate'],
                        capture_output=True,
                        timeout=30
                    )
                    scaling_config['current_workers'] = new_count
                    scaling_config['desired_workers'] = new_count
                
                # Scale up if utilization is high and we haven't reached max workers
                elif utilization > scaling_config['scale_up_threshold'] and scaling_config['current_workers'] < scaling_config['max_workers']:
                    new_count = min(scaling_config['current_workers'] + 1, scaling_config['max_workers'])
                    logger.info(f"📈 High utilization ({utilization:.1%}), scaling up to {new_count} workers")
                    
                    # Execute scale up
                    subprocess.run(
                        ['docker', 'compose', 'up', '-d', '--scale', f'worker={new_count}', '--no-recreate'],
                        capture_output=True,
                        timeout=30
                    )
                    scaling_config['current_workers'] = new_count
                    scaling_config['desired_workers'] = new_count
            
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            logger.error(f"Error in auto-shrink monitor: {str(e)}")
            time.sleep(10)

def simulate_metrics():
    """Simulate some metrics for demonstration"""
    # This would be removed in production
    def update_metrics():
        while True:
            # Simulate random active jobs and workers
            active_jobs.set(random.randint(0, 10))
            active_workers.set(random.randint(2, 8))
            time.sleep(10)
    
    thread = threading.Thread(target=update_metrics, daemon=True)
    thread.start()

@app.route('/api/v1/jobs/<job_id>/logs', methods=['GET'])
def get_job_logs(job_id):
    """Get or stream logs for a specific job (fallback endpoint)"""
    # Unknown jobs are not auto-registered with fake data
    if job_id not in jobs_data:
        return jsonify({'error': 'Job not found', 'job_id': job_id}), 404

    job = jobs_data[job_id]
    
    # If client wants SSE streaming
    if request.headers.get('Accept') == 'text/event-stream':
        def generate_logs():
            # Send existing logs first
            existing_logs = job.get('logs', [])
            for log in existing_logs:
                yield f"data: {log}\n\n"
                time.sleep(0.1)
            
            # Generate some demo logs if none exist
            if not existing_logs:
                demo_logs = [
                    f"[{time.strftime('%H:%M:%S')}] INFO: Job {job_id} initialized",
                    f"[{time.strftime('%H:%M:%S')}] INFO: Starting training process",
                    f"[{time.strftime('%H:%M:%S')}] INFO: Loading dataset",
                    f"[{time.strftime('%H:%M:%S')}] INFO: Model configuration complete",
                ]
                
                job_status = job.get('status', 'UNKNOWN')
                if job_status == 'COMPLETED':
                    demo_logs.extend([
                        f"[{time.strftime('%H:%M:%S')}] INFO: Training completed successfully",
                        f"[{time.strftime('%H:%M:%S')}] INFO: Final accuracy: {job.get('metrics', {}).get('accuracy', 0.85):.3f}",
                        f"[{time.strftime('%H:%M:%S')}] INFO: Log streaming ended"
                    ])
                elif job_status == 'FAILED':
                    demo_logs.extend([
                        f"[{time.strftime('%H:%M:%S')}] ERROR: Training failed",
                        f"[{time.strftime('%H:%M:%S')}] INFO: Log streaming ended"
                    ])
                else:
                    demo_logs.extend([
                        f"[{time.strftime('%H:%M:%S')}] INFO: Training in progress...",
                        f"[{time.strftime('%H:%M:%S')}] INFO: Current epoch: {job.get('progress', {}).get('current_epoch', 1)}"
                    ])
                
                for log in demo_logs:
                    yield f"data: {log}\n\n"
                    time.sleep(0.2)
        
        return Response(
            generate_logs(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
    else:
        # Return logs as JSON
        return jsonify({
            'job_id': job_id,
            'logs': job.get('logs', []),
            'status': job.get('status', 'UNKNOWN')
        }), 200

if __name__ == '__main__':
    # simulate_metrics()  # Uncomment for demo mode
    port = int(os.getenv('PORT', 8082))
    logger.info(f"Starting monitoring service on port {port}")
    app.run(host='0.0.0.0', port=port)
