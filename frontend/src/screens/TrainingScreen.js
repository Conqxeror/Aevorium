import React, { useEffect, useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, Alert, RefreshControl } from 'react-native';
import { Button, Text, TextInput } from 'react-native-paper';
import api from '../components/ApiClient';
import GradientHeader from '../components/GradientHeader';
import SectionCard from '../components/SectionCard';
import MetricCard from '../components/MetricCard';
import StatusBadge from '../components/StatusBadge';
import ActivityItem from '../components/ActivityItem';
import { Colors, Spacing } from '../theme';

export default function TrainingScreen() {
  const [rounds, setRounds] = useState('3');
  const [numClients, setNumClients] = useState('2');
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    await Promise.all([
      fetchTrainingStatus(),
      loadTrainingHistory(),
      loadPrivacyInfo(),
    ]);
  };

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  }, []);

  const fetchTrainingStatus = async () => {
    try {
      const r = await api.get('/training-status');
      setTrainingStatus(r.data);
    } catch (e) {
      // Endpoint might not exist in older API versions
      setTrainingStatus({ status: 'unknown' });
    }
  };

  const loadTrainingHistory = async () => {
    try {
      const r = await api.get('/audit-log');
      const logs = r.data || [];
      const trainLogs = logs
        .filter((l) => l.event_type?.startsWith('TRAINING') || l.event_type?.includes('privacy'))
        .slice(-20);
      setTrainingHistory(trainLogs);
    } catch (e) {
      setTrainingHistory([]);
    }
  };

  const loadPrivacyInfo = async () => {
    try {
      const r = await api.get('/privacy-budget');
      setModelInfo({
        rounds: r.data?.num_rounds || 0,
        epsilon: r.data?.cumulative_epsilon || 0,
      });
    } catch (e) {
      setModelInfo(null);
    }
  };

  const startTraining = async () => {
    const numRounds = Number(rounds);
    const clients = Number(numClients);

    if (!numRounds || numRounds < 1 || numRounds > 100) {
      Alert.alert('Invalid rounds', 'Must be between 1 and 100');
      return;
    }

    if (!clients || clients < 2 || clients > 10) {
      Alert.alert('Invalid clients', 'Must be between 2 and 10');
      return;
    }

    Alert.alert(
      'Start Training',
      `This will start federated training with ${numRounds} rounds and ${clients} clients. Continue?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Start',
          onPress: async () => {
            setLoading(true);
            try {
              const r = await api.post('/train', {
                rounds: numRounds,
                num_clients: clients,
              });
              if (r.status === 200) {
                Alert.alert('Training Started', 'Training is running in the background. Check status periodically.');
                loadData();
              } else {
                Alert.alert('Failed', `Status ${r.status}`);
              }
            } catch (e) {
              Alert.alert('Error', e.message || 'Failed to start training');
            } finally {
              setLoading(false);
            }
          },
        },
      ]
    );
  };

  const getStatusBadge = () => {
    if (!trainingStatus) return { status: 'info', label: 'Unknown' };
    switch (trainingStatus.status) {
      case 'running':
        return { status: 'warning', label: 'Training Running' };
      case 'completed':
        return { status: 'success', label: 'Training Completed' };
      case 'failed':
        return { status: 'error', label: 'Training Failed' };
      case 'idle':
        return { status: 'info', label: 'Ready to Train' };
      default:
        return { status: 'info', label: 'Unknown' };
    }
  };

  const statusBadge = getStatusBadge();

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.scrollContent}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={Colors.primary}
          colors={[Colors.primary]}
        />
      }
    >
      <GradientHeader
        title="Training"
        subtitle="Federated Learning Control"
        icon="brain"
      />

      <View style={styles.content}>
        {/* Status */}
        <View style={styles.statusContainer}>
          <StatusBadge status={statusBadge.status} label={statusBadge.label} />
        </View>

        {/* Metrics */}
        <View style={styles.metricsRow}>
          <MetricCard
            title="Completed Rounds"
            value={modelInfo?.rounds?.toString() || '0'}
            icon="counter"
            iconColor={Colors.chartBlue}
            accentColor={Colors.chartBlue}
          />
          <MetricCard
            title="Total ε Spent"
            value={modelInfo?.epsilon?.toFixed(2) || '0.00'}
            icon="shield-lock"
            iconColor={Colors.chartPurple}
            accentColor={Colors.chartPurple}
          />
        </View>

        {/* Training Configuration */}
        <SectionCard title="Start New Training" icon="play-circle">
          <Text style={styles.label}>Number of Rounds</Text>
          <TextInput
            mode="outlined"
            value={rounds}
            onChangeText={setRounds}
            keyboardType="numeric"
            style={styles.input}
            outlineColor={Colors.border}
            activeOutlineColor={Colors.primary}
            textColor={Colors.text}
          />
          <Text style={styles.hint}>1-100 rounds per training session</Text>

          <Text style={[styles.label, { marginTop: Spacing.md }]}>Number of Clients</Text>
          <TextInput
            mode="outlined"
            value={numClients}
            onChangeText={setNumClients}
            keyboardType="numeric"
            style={styles.input}
            outlineColor={Colors.border}
            activeOutlineColor={Colors.primary}
            textColor={Colors.text}
          />
          <Text style={styles.hint}>2-10 federated clients</Text>

          <Button
            mode="contained"
            onPress={startTraining}
            loading={loading}
            disabled={loading || trainingStatus?.status === 'running'}
            style={styles.button}
            icon="rocket-launch"
            buttonColor={Colors.secondary}
          >
            {loading ? 'Starting...' : 'Start Federated Training'}
          </Button>

          {trainingStatus?.status === 'running' && (
            <Text style={styles.runningText}>
              ⏳ Training is currently running. Please wait for it to complete.
            </Text>
          )}
        </SectionCard>

        {/* Training Info */}
        <SectionCard title="About Federated Training" icon="information" collapsible defaultExpanded={false}>
          <Text style={styles.infoText}>
            Federated Learning trains the diffusion model across distributed data sources without
            centralizing raw data. Each client trains locally with differential privacy (DP-SGD)
            and only shares encrypted model weights.
          </Text>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Framework:</Text>
            <Text style={styles.infoValue}>Flower (FL)</Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Privacy:</Text>
            <Text style={styles.infoValue}>Opacus DP-SGD</Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Aggregation:</Text>
            <Text style={styles.infoValue}>FedAvg</Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Model:</Text>
            <Text style={styles.infoValue}>Tabular Diffusion</Text>
          </View>
        </SectionCard>

        {/* Training History */}
        <SectionCard title="Training Events" icon="history" collapsible defaultExpanded={true}>
          {trainingHistory.length > 0 ? (
            trainingHistory.slice(-10).reverse().map((log, i) => (
              <ActivityItem
                key={i}
                timestamp={log.timestamp}
                eventType={log.event_type}
                details={log.details}
              />
            ))
          ) : (
            <Text style={styles.emptyText}>No training events yet.</Text>
          )}
        </SectionCard>

        <View style={styles.bottomSpacer} />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  scrollContent: {
    padding: 0,
  },
  content: {
    padding: Spacing.md,
    marginTop: -Spacing.xl,
  },
  statusContainer: {
    marginBottom: Spacing.md,
  },
  metricsRow: {
    flexDirection: 'row',
    marginHorizontal: -Spacing.xs,
    marginBottom: Spacing.sm,
  },
  label: {
    fontSize: 14,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
    fontWeight: '500',
  },
  hint: {
    fontSize: 12,
    color: Colors.textMuted,
    marginTop: Spacing.xs,
  },
  input: {
    backgroundColor: Colors.surface,
  },
  button: {
    marginTop: Spacing.lg,
  },
  runningText: {
    fontSize: 14,
    color: Colors.warning,
    marginTop: Spacing.md,
    textAlign: 'center',
  },
  infoText: {
    fontSize: 14,
    color: Colors.textSecondary,
    lineHeight: 20,
    marginBottom: Spacing.md,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: Spacing.xs,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  infoLabel: {
    fontSize: 14,
    color: Colors.textMuted,
  },
  infoValue: {
    fontSize: 14,
    color: Colors.text,
    fontWeight: '500',
  },
  emptyText: {
    color: Colors.textMuted,
    fontSize: 14,
    textAlign: 'center',
    paddingVertical: Spacing.md,
  },
  bottomSpacer: {
    height: Spacing.xl,
  },
});
