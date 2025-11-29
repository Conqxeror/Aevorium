import React, { useEffect, useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, RefreshControl, Alert } from 'react-native';
import { Text, ActivityIndicator, TextInput, Button } from 'react-native-paper';
import api, { setApiToken, getApiBaseUrl, apiMethods } from '../components/ApiClient';
import GradientHeader from '../components/GradientHeader';
import MetricCard from '../components/MetricCard';
import SectionCard from '../components/SectionCard';
import StatusBadge from '../components/StatusBadge';
import ActivityItem from '../components/ActivityItem';
import { Colors, Spacing, FontSizes, BorderRadius } from '../theme';

export default function OverviewScreen() {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [apiStatus, setApiStatus] = useState({ online: false, message: 'Checking...' });
  const [privacy, setPrivacy] = useState(null);
  const [logs, setLogs] = useState([]);
  const [apiToken, setApiTokenField] = useState('');
  const [trainingStatus, setTrainingStatus] = useState(null);

  const loadData = useCallback(async () => {
    await Promise.all([
      checkApiHealth(),
      loadPrivacy(),
      loadLogs(),
      loadTrainingStatus(),
    ]);
    setLoading(false);
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    setApiToken(apiToken);
  }, [apiToken]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  }, [loadData]);

  const checkApiHealth = async () => {
    try {
      const r = await apiMethods.healthCheck();
      if (r.status === 200) {
        setApiStatus({ online: true, message: 'Connected', latency: r.data?.latency });
      }
    } catch (e) {
      setApiStatus({ online: false, message: e.message || 'Connection failed' });
    }
  };

  const loadPrivacy = async () => {
    try {
      const r = await apiMethods.getPrivacyBudget();
      setPrivacy(r.data);
    } catch (e) {
      setPrivacy(null);
    }
  };

  const loadLogs = async () => {
    try {
      const r = await apiMethods.getAuditLog();
      setLogs(r.data || []);
    } catch (e) {
      setLogs([]);
    }
  };

  const loadTrainingStatus = async () => {
    try {
      const r = await apiMethods.getTrainingStatus();
      setTrainingStatus(r.data);
    } catch (e) {
      setTrainingStatus({ status: 'unknown' });
    }
  };

  const formatNumber = (num) => {
    if (num === null || num === undefined) return '–';
    if (num === Infinity || num === 'Infinity') return '∞';
    if (typeof num === 'number') return num.toFixed(2);
    return String(num);
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
        <Text style={styles.loadingText}>Loading dashboard...</Text>
      </View>
    );
  }

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
        title="Aevorium"
        subtitle="Federated Learning for Synthetic Healthcare Data"
        icon="brain"
      />

      <View style={styles.content}>
        {/* API Status */}
        <SectionCard title="System Status" icon="server">
          <View style={styles.statusRow}>
            <View style={styles.statusInfo}>
              <Text style={styles.statusLabel}>API Connection</Text>
              <Text style={styles.apiUrl}>{getApiBaseUrl()}</Text>
            </View>
            <StatusBadge
              status={apiStatus.online ? 'success' : 'error'}
              label={apiStatus.online ? 'Online' : 'Offline'}
            />
          </View>
          {apiStatus.online && (
            <View style={styles.tokenInputContainer}>
              <TextInput
                mode="outlined"
                label="API Token (optional)"
                placeholder="Enter Bearer token"
                value={apiToken}
                onChangeText={setApiTokenField}
                style={styles.tokenInput}
                outlineColor={Colors.border}
                activeOutlineColor={Colors.primary}
                textColor={Colors.text}
                secureTextEntry
                right={<TextInput.Icon icon="key" color={Colors.textMuted} />}
              />
            </View>
          )}
        </SectionCard>

        {/* Key Metrics */}
        <View style={styles.metricsGrid}>
          <View style={styles.metricsRow}>
            <MetricCard
              title="Privacy Budget (ε)"
              value={formatNumber(privacy?.cumulative_epsilon)}
              icon="shield-lock"
              iconColor={Colors.chartPurple}
              accentColor={Colors.chartPurple}
              subtitle={`of ${formatNumber(privacy?.max_epsilon)} max`}
            />
            <MetricCard
              title="Budget Remaining"
              value={formatNumber(privacy?.budget_remaining)}
              icon="battery-charging"
              iconColor={Colors.chartGreen}
              accentColor={Colors.chartGreen}
              trendDirection={privacy?.budget_remaining > 5 ? 'up' : 'down'}
            />
          </View>
          <View style={styles.metricsRow}>
            <MetricCard
              title="Training Rounds"
              value={String(privacy?.num_rounds || 0)}
              icon="repeat"
              iconColor={Colors.chartBlue}
              accentColor={Colors.chartBlue}
            />
            <MetricCard
              title="Training Status"
              value={trainingStatus?.status || 'Unknown'}
              icon="brain"
              iconColor={Colors.chartOrange}
              accentColor={Colors.chartOrange}
            />
          </View>
        </View>

        {/* Recent Activity */}
        <SectionCard title="Recent Activity" icon="history" collapsible defaultExpanded={true}>
          {logs && logs.length > 0 ? (
            logs.slice(-7).reverse().map((log, index) => (
              <ActivityItem
                key={index}
                timestamp={log.timestamp}
                eventType={log.event_type}
                details={log.details}
              />
            ))
          ) : (
            <View style={styles.emptyState}>
              <Text style={styles.emptyText}>No activity recorded yet</Text>
              <Text style={styles.emptySubtext}>
                Start training or generate data to see events here
              </Text>
            </View>
          )}
        </SectionCard>

        {/* Quick Actions */}
        <SectionCard title="Quick Actions" icon="lightning-bolt">
          <View style={styles.actionsRow}>
            <Button
              mode="contained"
              icon="refresh"
              onPress={onRefresh}
              style={styles.actionButton}
              buttonColor={Colors.primary}
            >
              Refresh
            </Button>
            <Button
              mode="outlined"
              icon="cog"
              onPress={() => Alert.alert('Settings', 'Settings panel coming soon')}
              style={styles.actionButton}
              textColor={Colors.text}
            >
              Settings
            </Button>
          </View>
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
    padding: 0, // Remove padding from scroll view content container
  },
  content: {
    padding: Spacing.md, // Add padding to the content wrapper
    marginTop: -Spacing.xl, // Pull content up to overlap with header slightly if desired, or just 0
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: Colors.background,
  },
  loadingText: {
    marginTop: Spacing.md,
    color: Colors.textSecondary,
    fontSize: FontSizes.md,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  statusInfo: {
    flex: 1,
  },
  statusLabel: {
    fontSize: FontSizes.md,
    color: Colors.text,
    fontWeight: '500',
  },
  apiUrl: {
    fontSize: FontSizes.sm,
    color: Colors.textMuted,
    marginTop: 2,
  },
  tokenInputContainer: {
    marginTop: Spacing.sm,
  },
  tokenInput: {
    backgroundColor: Colors.surface,
  },
  metricsGrid: {
    marginVertical: Spacing.sm,
  },
  metricsRow: {
    flexDirection: 'row',
    marginHorizontal: -Spacing.xs,
    marginBottom: Spacing.sm,
  },
  emptyState: {
    padding: Spacing.lg,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: FontSizes.md,
    color: Colors.textSecondary,
    fontWeight: '500',
  },
  emptySubtext: {
    fontSize: FontSizes.sm,
    color: Colors.textMuted,
    marginTop: Spacing.xs,
    textAlign: 'center',
  },
  actionsRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  actionButton: {
    flex: 1,
    borderRadius: BorderRadius.md,
  },
  bottomSpacer: {
    height: Spacing.xl,
  },
});
