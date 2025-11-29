import React, { useEffect, useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, RefreshControl, Alert, FlatList } from 'react-native';
import { Text, Searchbar, Chip, Button, ActivityIndicator, Divider } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import api, { apiMethods } from '../components/ApiClient';
import GradientHeader from '../components/GradientHeader';
import SectionCard from '../components/SectionCard';
import ActivityItem from '../components/ActivityItem';
import MetricCard from '../components/MetricCard';
import { Colors, Spacing, FontSizes, BorderRadius } from '../theme';

const EVENT_FILTERS = [
  { key: 'all', label: 'All', icon: 'format-list-bulleted' },
  { key: 'training', label: 'Training', icon: 'brain' },
  { key: 'generation', label: 'Generation', icon: 'creation' },
  { key: 'privacy', label: 'Privacy', icon: 'shield-lock' },
  { key: 'system', label: 'System', icon: 'cog' },
];

export default function AuditLogScreen() {
  const [logs, setLogs] = useState([]);
  const [filteredLogs, setFilteredLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all');
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    loadLogs();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [logs, searchQuery, activeFilter]);

  const loadLogs = async () => {
    try {
      const r = await apiMethods.getAuditLog();
      setLogs(r.data || []);
    } catch (e) {
      setLogs([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadLogs();
  }, []);

  const applyFilters = () => {
    let filtered = [...logs];

    // Apply event type filter
    if (activeFilter !== 'all') {
      filtered = filtered.filter((log) => {
        const eventType = (log.event_type || '').toLowerCase();
        switch (activeFilter) {
          case 'training':
            return eventType.includes('training') || eventType.includes('fl_');
          case 'generation':
            return eventType.includes('generat') || eventType.includes('sampl');
          case 'privacy':
            return eventType.includes('privacy') || eventType.includes('budget') || eventType.includes('epsilon');
          case 'system':
            return eventType.includes('system') || eventType.includes('error') || eventType.includes('health');
          default:
            return true;
        }
      });
    }

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((log) => {
        const eventType = (log.event_type || '').toLowerCase();
        const details = JSON.stringify(log.details || {}).toLowerCase();
        const timestamp = (log.timestamp || '').toLowerCase();
        return eventType.includes(query) || details.includes(query) || timestamp.includes(query);
      });
    }

    setFilteredLogs(filtered.reverse());
  };

  const clearHistory = () => {
    Alert.alert(
      'Clear History',
      'This will delete all audit logs and generated synthetic data. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear All',
          style: 'destructive',
          onPress: async () => {
            setClearing(true);
            try {
              await apiMethods.clearHistory();
              Alert.alert('Success', 'All history has been cleared');
              setLogs([]);
              setFilteredLogs([]);
            } catch (e) {
              Alert.alert('Error', e.response?.data?.detail || 'Failed to clear history');
            } finally {
              setClearing(false);
            }
          },
        },
      ]
    );
  };

  const getEventStats = () => {
    const stats = {
      total: logs.length,
      training: 0,
      generation: 0,
      privacy: 0,
    };

    logs.forEach((log) => {
      const eventType = (log.event_type || '').toLowerCase();
      if (eventType.includes('training') || eventType.includes('fl_')) stats.training++;
      if (eventType.includes('generat') || eventType.includes('sampl')) stats.generation++;
      if (eventType.includes('privacy') || eventType.includes('budget')) stats.privacy++;
    });

    return stats;
  };

  const stats = getEventStats();

  const renderLogItem = ({ item, index }) => (
    <ActivityItem
      key={index}
      timestamp={item.timestamp}
      eventType={item.event_type}
      details={item.details}
      showFullDetails
    />
  );

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
        <Text style={styles.loadingText}>Loading audit logs...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.contentContainer}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={Colors.primary}
            colors={[Colors.primary]}
          />
        }
        stickyHeaderIndices={[1]}
      >
        <GradientHeader
          title="Audit Log"
          subtitle="Security & compliance event tracking"
          icon="file-document"
        />

        {/* Search & Filters - Sticky */}
        <View style={styles.stickyHeader}>
          <Searchbar
            placeholder="Search events..."
            onChangeText={setSearchQuery}
            value={searchQuery}
            style={styles.searchBar}
            inputStyle={styles.searchInput}
            iconColor={Colors.textMuted}
            placeholderTextColor={Colors.textMuted}
          />

          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.filterScroll}
            contentContainerStyle={styles.filterContainer}
          >
            {EVENT_FILTERS.map((filter) => (
              <Chip
                key={filter.key}
                mode={activeFilter === filter.key ? 'flat' : 'outlined'}
                selected={activeFilter === filter.key}
                onPress={() => setActiveFilter(filter.key)}
                style={[
                  styles.filterChip,
                  activeFilter === filter.key && styles.filterChipSelected,
                ]}
                textStyle={styles.filterChipText}
                icon={() => (
                  <MaterialCommunityIcons
                    name={filter.icon}
                    size={16}
                    color={activeFilter === filter.key ? Colors.text : Colors.textSecondary}
                  />
                )}
              >
                {filter.label}
              </Chip>
            ))}
          </ScrollView>
        </View>

        {/* Stats */}
        <View style={styles.statsRow}>
          <MetricCard
            title="Total Events"
            value={String(stats.total)}
            icon="format-list-numbered"
            iconColor={Colors.chartBlue}
            accentColor={Colors.chartBlue}
          />
          <MetricCard
            title="Training"
            value={String(stats.training)}
            icon="brain"
            iconColor={Colors.chartOrange}
            accentColor={Colors.chartOrange}
          />
        </View>

        <View style={styles.statsRow}>
          <MetricCard
            title="Generation"
            value={String(stats.generation)}
            icon="creation"
            iconColor={Colors.chartGreen}
            accentColor={Colors.chartGreen}
          />
          <MetricCard
            title="Privacy"
            value={String(stats.privacy)}
            icon="shield-lock"
            iconColor={Colors.chartPurple}
            accentColor={Colors.chartPurple}
          />
        </View>

        {/* Clear History Button */}
        <View style={styles.clearSection}>
          <Button
            mode="outlined"
            onPress={clearHistory}
            loading={clearing}
            disabled={clearing || logs.length === 0}
            icon="delete-sweep"
            textColor={Colors.danger}
            style={styles.clearButton}
          >
            Clear All History
          </Button>
          <Text style={styles.clearHint}>
            Removes audit logs and generated data files
          </Text>
        </View>

        {/* Event List */}
        <SectionCard 
          title={`Events (${filteredLogs.length})`} 
          icon="history"
        >
          {filteredLogs.length > 0 ? (
            filteredLogs.slice(0, 50).map((log, index) => (
              <React.Fragment key={index}>
                <ActivityItem
                  timestamp={log.timestamp}
                  eventType={log.event_type}
                  details={log.details}
                  showFullDetails
                />
                {index < filteredLogs.length - 1 && <Divider style={styles.divider} />}
              </React.Fragment>
            ))
          ) : (
            <View style={styles.emptyState}>
              <MaterialCommunityIcons name="file-document-outline" size={48} color={Colors.textMuted} />
              <Text style={styles.emptyText}>
                {searchQuery || activeFilter !== 'all' ? 'No matching events' : 'No audit events yet'}
              </Text>
              <Text style={styles.emptySubtext}>
                {searchQuery || activeFilter !== 'all'
                  ? 'Try adjusting your filters'
                  : 'Events will appear here as actions are performed'}
              </Text>
            </View>
          )}

          {filteredLogs.length > 50 && (
            <Text style={styles.moreText}>
              Showing 50 of {filteredLogs.length} events
            </Text>
          )}
        </SectionCard>

        <View style={styles.bottomSpacer} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  scrollView: {
    flex: 1,
  },
  contentContainer: {
    paddingHorizontal: Spacing.md,
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
  stickyHeader: {
    backgroundColor: Colors.background,
    paddingVertical: Spacing.sm,
    marginHorizontal: -Spacing.md,
    paddingHorizontal: Spacing.md,
  },
  searchBar: {
    backgroundColor: Colors.surface,
    borderRadius: BorderRadius.lg,
    elevation: 0,
  },
  searchInput: {
    color: Colors.text,
  },
  filterScroll: {
    marginTop: Spacing.sm,
    marginHorizontal: -Spacing.md,
  },
  filterContainer: {
    paddingHorizontal: Spacing.md,
    gap: Spacing.sm,
  },
  filterChip: {
    backgroundColor: Colors.surfaceLight,
    borderColor: Colors.border,
  },
  filterChipSelected: {
    backgroundColor: Colors.primary,
  },
  filterChipText: {
    color: Colors.text,
    fontSize: FontSizes.sm,
  },
  statsRow: {
    flexDirection: 'row',
    marginHorizontal: -Spacing.xs,
    marginTop: Spacing.sm,
  },
  clearSection: {
    marginTop: Spacing.md,
    marginBottom: Spacing.sm,
    alignItems: 'center',
  },
  clearButton: {
    borderColor: Colors.danger,
    borderRadius: BorderRadius.md,
  },
  clearHint: {
    fontSize: FontSizes.xs,
    color: Colors.textMuted,
    marginTop: Spacing.xs,
  },
  divider: {
    backgroundColor: Colors.border,
    marginVertical: Spacing.sm,
  },
  emptyState: {
    alignItems: 'center',
    padding: Spacing.xl,
  },
  emptyText: {
    fontSize: FontSizes.md,
    color: Colors.textSecondary,
    fontWeight: '500',
    marginTop: Spacing.md,
  },
  emptySubtext: {
    fontSize: FontSizes.sm,
    color: Colors.textMuted,
    marginTop: Spacing.xs,
    textAlign: 'center',
  },
  moreText: {
    fontSize: FontSizes.sm,
    color: Colors.textMuted,
    textAlign: 'center',
    marginTop: Spacing.md,
    fontStyle: 'italic',
  },
  bottomSpacer: {
    height: Spacing.xl,
  },
});
