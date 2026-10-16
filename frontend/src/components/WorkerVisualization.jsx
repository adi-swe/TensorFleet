import React, { useState } from 'react';
import { monitoringAPI } from '../api/api';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  LinearProgress,
  Skeleton,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  Computer as ComputerIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
  Memory as MemoryIcon,
  Psychology as CpuIcon,
  AccessTime as AccessTimeIcon,
  Task as TaskIcon,
  Circle as CircleIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { usePolling } from '../hooks/usePolling';
import { monoFontFamily } from '../theme/typography';

// 42px icon badge + overline label + h3 stat number, per the design system.
const StatCard = ({ title, subtitle, value, icon: Icon, color = 'primary', loading }) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 1 }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
            {title}
          </Typography>
          {loading ? (
            <Skeleton width={72} height={40} />
          ) : (
            <Typography variant="h3" sx={{ fontWeight: 700, lineHeight: 1.15 }}>
              {value ?? '—'}
            </Typography>
          )}
          {subtitle && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }} noWrap>
              {subtitle}
            </Typography>
          )}
        </Box>
        <Box
          sx={(theme) => {
            const main = color === 'grey' ? theme.palette.grey[500] : theme.palette[color].main;
            return {
              width: 42,
              height: 42,
              borderRadius: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              backgroundColor: alpha(main, theme.palette.mode === 'dark' ? 0.16 : 0.1),
              color: color === 'grey' ? theme.palette.text.secondary : main,
            };
          }}
        >
          <Icon fontSize="small" />
        </Box>
      </Box>
    </CardContent>
  </Card>
);

const WorkerVisualization = () => {
  const [workers, setWorkers] = useState([]);
  const [workerStats, setWorkerStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  const fetchWorkerActivity = async () => {
    try {
      const response = await monitoringAPI.getWorkerActivity();
      const workersData = response.data.workers || [];
      setWorkers(workersData);

      // Calculate average resource usage
      const activeWorkers = workersData.filter(w => w.is_active);
      const averageMeasured = (key) => {
        const measured = activeWorkers.map(worker => worker[key]).filter(value => Number.isFinite(value) && value >= 0);
        return measured.length ? Math.round(measured.reduce((sum, value) => sum + value, 0) / measured.length) : null;
      };
      const avgCpu = averageMeasured('cpu_usage');
      const avgMemory = averageMeasured('memory_usage');

      setWorkerStats({
        total: response.data.total_workers || workersData.length,
        active: response.data.active_workers || workersData.filter(w => w.is_active).length,
        busy: response.data.busy_workers || workersData.filter(w => w.status === 'BUSY').length,
        idle: workersData.filter(w => w.status === 'IDLE').length,
        offline: workersData.filter(w => w.status === 'OFFLINE').length,
        avgCpu,
        avgMemory
      });
      setLastUpdate(new Date());
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch worker activity:', error);
      setLoading(false);
    }
  };

  usePolling(fetchWorkerActivity, 5000);

  // BUSY→success, IDLE→grey, OFFLINE/ERROR→error.
  const getStatusColor = (status) => {
    switch (status) {
      case 'BUSY':
        return 'success';
      case 'IDLE':
        return 'default';
      case 'OFFLINE':
      case 'ERROR':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status, isActive) => {
    if (!isActive) return <CircleIcon sx={{ fontSize: 12, color: 'error.main' }} />;

    switch (status) {
      case 'BUSY':
        return <CircleIcon sx={{ fontSize: 12, color: 'success.main', animation: 'pulse 2s infinite' }} />;
      case 'IDLE':
        return <CircleIcon sx={{ fontSize: 12, color: 'text.secondary' }} />;
      default:
        return <ComputerIcon fontSize="small" sx={{ color: 'text.secondary' }} />;
    }
  };

  const usageLevel = (value) => ((value || 0) > 80 ? 'error' : (value || 0) > 60 ? 'warning' : 'success');

  const formatWorkerId = (workerId, index) => {
    // Convert hex UUID to user-friendly name like worker-1, worker-2, etc.
    if (!workerId) return `Worker ${index + 1}`;

    // If it's already in a friendly format, return as-is
    if (workerId.startsWith('worker-') && !workerId.includes('-') !== 4) {
      return workerId;
    }

    // Create a consistent hash from the worker ID to ensure same worker gets same number
    let hash = 0;
    for (let i = 0; i < workerId.length; i++) {
      const char = workerId.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }

    // Use absolute value and mod to get a consistent worker number
    const workerNumber = Math.abs(hash % 100) + 1; // 1-100 range
    return `worker-${workerNumber}`;
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
  };

  const formatUptime = (seconds) => {
    if (!seconds || seconds < 0) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3, flexWrap: 'wrap' }}>
        <ComputerIcon fontSize="small" color="primary" />
        <Typography variant="h5">Live Worker Activity Monitor</Typography>
        <Chip size="small" label={`${workerStats.total || 0} Workers`} color="primary" />
        <Chip size="small" label="Live" color="success" />
      </Box>

      {/* Summary Stats Row */}
      <Grid container spacing={2.5} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2.4}>
          <StatCard
            title="Total"
            subtitle="Workers"
            value={workerStats.total || 0}
            icon={ComputerIcon}
            color="primary"
            loading={loading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <StatCard
            title="Busy"
            subtitle="Processing"
            value={workerStats.busy || 0}
            icon={SpeedIcon}
            color="success"
            loading={loading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <StatCard
            title="Idle"
            subtitle="Available"
            value={workerStats.idle || 0}
            icon={ScheduleIcon}
            color="grey"
            loading={loading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <StatCard
            title="Avg CPU"
            subtitle="Across active workers"
            value={Number.isFinite(workerStats.avgCpu) ? `${workerStats.avgCpu}%` : 'Not measured'}
            icon={CpuIcon}
            color="info"
            loading={loading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <StatCard
            title="Avg Memory"
            subtitle="Across active workers"
            value={Number.isFinite(workerStats.avgMemory) ? `${workerStats.avgMemory}%` : 'Not measured'}
            icon={MemoryIcon}
            color="secondary"
            loading={loading}
          />
        </Grid>
      </Grid>

      <Grid container spacing={2.5}>
        {workers.map((worker, index) => (
          <Grid item xs={12} md={6} lg={4} key={worker.worker_id || index}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  {getStatusIcon(worker.status, worker.is_active)}
                  <Typography
                    variant="h6"
                    sx={{
                      color: worker.is_active ? 'text.primary' : 'text.disabled',
                      fontFamily: monoFontFamily,
                      flex: 1,
                    }}
                  >
                    {formatWorkerId(worker.worker_id, index)}
                  </Typography>
                  <Chip
                    size="small"
                    label={worker.status}
                    color={getStatusColor(worker.status)}
                  />
                </Box>

                <List dense>
                  {worker.current_task_id && (
                    <ListItem sx={{ px: 0 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <TaskIcon fontSize="small" color="info" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Current Task"
                        secondary={worker.current_task_id}
                        secondaryTypographyProps={{
                          sx: { wordBreak: 'break-all', fontSize: '0.75rem', fontFamily: monoFontFamily }
                        }}
                      />
                    </ListItem>
                  )}

                  <ListItem sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 30 }}>
                      <CheckCircleIcon fontSize="small" color="success" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Tasks Completed"
                      secondary={worker.tasks_completed || 0}
                    />
                  </ListItem>

                  <ListItem sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 30 }}>
                      <AccessTimeIcon fontSize="small" color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Uptime"
                      secondary={formatUptime(worker.uptime)}
                    />
                  </ListItem>

                  <ListItem sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 30 }}>
                      <CpuIcon fontSize="small" color="secondary" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Resources"
                      secondary={
                        <Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                            <Typography variant="caption" component="div">
                              CPU
                            </Typography>
                            <Typography
                              variant="caption"
                              component="div"
                              sx={{ fontWeight: 700, color: Number.isFinite(worker.cpu_usage) && worker.cpu_usage >= 0 ? `${usageLevel(worker.cpu_usage)}.main` : 'text.secondary' }}
                            >
                              {Number.isFinite(worker.cpu_usage) && worker.cpu_usage >= 0 ? `${worker.cpu_usage}%` : 'Not measured'}
                            </Typography>
                          </Box>
                          {Number.isFinite(worker.cpu_usage) && worker.cpu_usage >= 0 && <LinearProgress
                            variant="determinate"
                            value={worker.cpu_usage}
                            color={usageLevel(worker.cpu_usage)}
                            sx={{ mb: 1 }}
                          />}
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                            <Typography variant="caption" component="div">
                              Memory
                            </Typography>
                            <Typography
                              variant="caption"
                              component="div"
                              sx={{ fontWeight: 700, color: Number.isFinite(worker.memory_usage) && worker.memory_usage >= 0 ? `${usageLevel(worker.memory_usage)}.main` : 'text.secondary' }}
                            >
                              {Number.isFinite(worker.memory_usage) && worker.memory_usage >= 0 ? `${worker.memory_usage}%` : 'Not measured'}
                            </Typography>
                          </Box>
                          {Number.isFinite(worker.memory_usage) && worker.memory_usage >= 0 && <LinearProgress
                            variant="determinate"
                            value={worker.memory_usage}
                            color={usageLevel(worker.memory_usage)}
                          />}
                        </Box>
                      }
                    />
                  </ListItem>
                </List>

                <Divider sx={{ my: 1 }} />
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                  Last Activity: {formatTimestamp(worker.last_activity_time)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}

        {workers.length === 0 && !loading && (
          <Grid item xs={12}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <ComputerIcon sx={{ fontSize: 40, color: 'text.disabled', mb: 1.5 }} />
                <Typography variant="body2" color="text.secondary">
                  No workers active — they will appear here when they register with the system
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Status Footer */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, flexWrap: 'wrap' }}>
            <Box
              sx={(theme) => ({
                width: 9,
                height: 9,
                borderRadius: '50%',
                backgroundColor: loading ? theme.palette.warning.main : theme.palette.success.main,
                animation: 'pulse 2.5s ease-in-out infinite',
              })}
            />
            <Typography variant="body2" color="text.secondary">
              {loading
                ? 'Connecting to fleet…'
                : `Live · updated ${lastUpdate.toLocaleTimeString()} · auto-refreshes every 5s`}
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default WorkerVisualization;
