import React, { useEffect, useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, Alert, RefreshControl, Dimensions } from 'react-native';
import { Text, Button, TextInput, Chip, Card, Divider } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LineChart } from 'react-native-chart-kit';
import api, { apiMethods } from '../components/ApiClient';
import GradientHeader from '../components/GradientHeader';
import SectionCard from '../components/SectionCard';
import MetricCard from '../components/MetricCard';
import ProgressGauge from '../components/ProgressGauge';
import { Colors, Spacing, FontSizes, BorderRadius } from '../theme';

const screenWidth = Dimensions.get('window').width;

export default function PrivacyScreen() {
  const [summary, setSummary] = useState(null);
  const [newBudget, setNewBudget] = useState('100');
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [roundHistory, setRoundHistory] = useState([]);

  useEffect(() => {
    fetchSummary();
    fetchRoundHistory();
  }, []);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await Promise.all([fetchSummary(), fetchRoundHistory()]);
    setRefreshing(false);
  }, []);

  const fetchSummary = async () => {
    try {
      const r = await apiMethods.getPrivacyBudget();
      setSummary(r.data);
    } catch (e) {
      setSummary(null);
    }
  };

  const fetchRoundHistory = async () => {
    try {
      const r = await apiMethods.getAuditLog();
      const logs = r.data || [];
      // Extract epsilon values from training rounds
      const trainingLogs = logs
        .filter((l) => l.event_type?.includes('privacy') || l.event_type?.includes('TRAINING'))
        .map((l, i) => ({
          round: i + 1,
          epsilon: l.details?.epsilon || l.details?.cumulative_epsilon || 0,
        }))
        .slice(-10);
      setRoundHistory(trainingLogs);
    } catch (e) {
      setRoundHistory([]);
    }
  };

  const setBudget = async () => {
    try {
      const budget = Number(newBudget);
      if (isNaN(budget) || budget <= 0) {
        Alert.alert('Invalid budget', 'Enter a valid positive epsilon value');
        return;
      }
      setLoading(true);
      const r = await api.post('/privacy-budget/set-limit', { total_budget: budget });
      if (r.status === 200) {
        Alert.alert('Success', 'Privacy budget limit updated');
        fetchSummary();
      } else {
        Alert.alert('Failed', `Status ${r.status}`);
      }
    } catch (e) {
      Alert.alert('Error', e.response?.data?.detail || 'Failed to set budget');
    } finally {
      setLoading(false);
    }
  };

  const resetBudget = async () => {
    Alert.alert(
      'Reset Privacy Budget',
      'This will reset the cumulative epsilon to 0. Continue?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: async () => {
            try {
              setLoading(true);
              const r = await api.post('/privacy-budget/reset');
              if (r.status === 200) {
                Alert.alert('Success', 'Privacy budget has been reset');
                fetchSummary();
              } else {
                Alert.alert('Failed', `Status ${r.status}`);
              }
            } catch (e) {
              Alert.alert('Error', e.response?.data?.detail || 'Failed to reset');
            } finally {
              setLoading(false);
            }
          },
        },
      ]
    );
  };

  const formatNumber = (num) => {
    if (num === null || num === undefined) return '–';
    if (num === Infinity || num === 'Infinity' || num === 'inf') return '∞';
    if (typeof num === 'number') {
      if (Number.isFinite(num)) return num.toFixed(2);
      return '∞';
    }
    return String(num);
  };

  const getBudgetUsagePercent = () => {
    if (!summary) return 0;
    const max = summary.max_epsilon || summary.total_budget;
    if (!max || max === Infinity || max === 'Infinity') return 0;
    const used = summary.cumulative_epsilon || 0;
    return Math.min((used / max) * 100, 100);
  };

  const getChartData = () => {
    if (roundHistory.length === 0) {
      return {
        labels: ['1', '2', '3', '4', '5'],
        datasets: [{ data: [0, 0, 0, 0, 0] }],
      };
    }
    return {
      labels: roundHistory.map((r) => String(r.round)),
      datasets: [{ data: roundHistory.map((r) => r.epsilon || 0) }],
    };
  };

  const budgetPresets = [10, 50, 100, 500];

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
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
        title="Privacy Budget"
        subtitle="Differential Privacy (ε, δ) Management"
        icon="shield-lock"
      />

      {/* Budget Overview */}
      <SectionCard title="Budget Overview" icon="gauge">
        <View style={styles.gaugeContainer}>
          <ProgressGauge
            value={getBudgetUsagePercent()}
            maxValue={100}
            size={140}
            strokeWidth={12}
            color={getBudgetUsagePercent() > 80 ? Colors.danger : Colors.primary}
            label={`${getBudgetUsagePercent().toFixed(0)}%`}
            sublabel="Budget Used"
          />
        </View>

        <View style={styles.metricsRow}>
          <MetricCard
            title="ε Spent"
            value={formatNumber(summary?.cumulative_epsilon)}
            icon="trending-up"
            iconColor={Colors.chartOrange}
            accentColor={Colors.chartOrange}
          />
          <MetricCard
            title="ε Remaining"
            value={formatNumber(summary?.budget_remaining)}
            icon="battery-charging"
            iconColor={Colors.chartGreen}
            accentColor={Colors.chartGreen}
          />
        </View>

        <View style={styles.metricsRow}>
          <MetricCard
            title="Max Budget"
            value={formatNumber(summary?.max_epsilon || summary?.total_budget)}
            icon="shield-check"
            iconColor={Colors.chartBlue}
            accentColor={Colors.chartBlue}
          />
          <MetricCard
            title="Delta (δ)"
            value={summary?.delta ? summary.delta.toExponential(0) : '1e-5'}
            icon="function-variant"
            iconColor={Colors.chartPurple}
            accentColor={Colors.chartPurple}
          />
        </View>
      </SectionCard>

      {/* Epsilon History Chart */}
      <SectionCard title="ε Accumulation Over Rounds" icon="chart-line">
        <LineChart
          data={getChartData()}
          width={screenWidth - 64}
          height={200}
          chartConfig={{
            backgroundColor: Colors.cardBackground,
            backgroundGradientFrom: Colors.cardBackground,
            backgroundGradientTo: Colors.surface,
            decimalPlaces: 2,
            color: (opacity = 1) => `rgba(31, 119, 180, ${opacity})`,
            labelColor: () => Colors.textSecondary,
            style: { borderRadius: BorderRadius.lg },
            propsForDots: {
              r: '4',
              strokeWidth: '2',
              stroke: Colors.primary,
            },
          }}
          bezier
          style={styles.chart}
        />
        <Text style={styles.chartNote}>
          Shows cumulative privacy loss per training round
        </Text>
      </SectionCard>

      {/* Budget Controls */}
      <SectionCard title="Budget Controls" icon="tune">
        <Text style={styles.label}>Set Maximum ε Budget</Text>
        <TextInput
          mode="outlined"
          value={newBudget}
          onChangeText={setNewBudget}
          keyboardType="numeric"
          style={styles.input}
          outlineColor={Colors.border}
          activeOutlineColor={Colors.primary}
          textColor={Colors.text}
          right={<TextInput.Icon icon="epsilon" color={Colors.textMuted} />}
        />

        <View style={styles.presetRow}>
          {budgetPresets.map((preset) => (
            <Chip
              key={preset}
              mode={newBudget === String(preset) ? 'flat' : 'outlined'}
              selected={newBudget === String(preset)}
              onPress={() => setNewBudget(String(preset))}
              style={[
                styles.presetChip,
                newBudget === String(preset) && styles.presetChipSelected,
              ]}
              textStyle={styles.presetChipText}
            >
              ε = {preset}
            </Chip>
          ))}
        </View>

        <View style={styles.buttonRow}>
          <Button
            mode="contained"
            onPress={setBudget}
            loading={loading}
            disabled={loading}
            style={styles.actionButton}
            icon="content-save"
            buttonColor={Colors.primary}
          >
            Set Budget
          </Button>
          <Button
            mode="outlined"
            onPress={resetBudget}
            disabled={loading}
            style={styles.actionButton}
            icon="refresh"
            textColor={Colors.danger}
          >
            Reset
          </Button>
        </View>
      </SectionCard>

      {/* Privacy Info */}
      <SectionCard title="About Differential Privacy" icon="information" collapsible defaultExpanded={false}>
        <View style={styles.infoItem}>
          <MaterialCommunityIcons name="epsilon" size={20} color={Colors.primary} />
          <View style={styles.infoContent}>
            <Text style={styles.infoTitle}>Epsilon (ε)</Text>
            <Text style={styles.infoText}>
              Privacy loss parameter. Lower values = stronger privacy. Typical range: 0.1 - 10.
            </Text>
          </View>
        </View>
        <Divider style={styles.divider} />
        <View style={styles.infoItem}>
          <MaterialCommunityIcons name="delta" size={20} color={Colors.chartPurple} />
          <View style={styles.infoContent}>
            <Text style={styles.infoTitle}>Delta (δ)</Text>
            <Text style={styles.infoText}>
              Probability of privacy breach. Should be very small (e.g., 1e-5).
            </Text>
          </View>
        </View>
        <Divider style={styles.divider} />
        <View style={styles.infoItem}>
          <MaterialCommunityIcons name="shield-lock" size={20} color={Colors.chartGreen} />
          <View style={styles.infoContent}>
            <Text style={styles.infoTitle}>Budget Tracking</Text>
            <Text style={styles.infoText}>
              Total ε accumulates with each training round. Once budget is exhausted, 
              no more training is allowed to protect data privacy.
            </Text>
          </View>
        </View>
      </SectionCard>

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  contentContainer: {
    padding: Spacing.md,
  },
  gaugeContainer: {
    alignItems: 'center',
    marginVertical: Spacing.md,
  },
  metricsRow: {
    flexDirection: 'row',
    marginHorizontal: -Spacing.xs,
    marginBottom: Spacing.sm,
  },
  chart: {
    borderRadius: BorderRadius.lg,
    marginVertical: Spacing.sm,
  },
  chartNote: {
    fontSize: FontSizes.xs,
    color: Colors.textMuted,
    textAlign: 'center',
    marginTop: Spacing.xs,
  },
  label: {
    fontSize: FontSizes.md,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
    fontWeight: '500',
  },
  input: {
    backgroundColor: Colors.surface,
    marginBottom: Spacing.sm,
  },
  presetRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
    marginBottom: Spacing.md,
  },
  presetChip: {
    backgroundColor: Colors.surfaceLight,
    borderColor: Colors.border,
  },
  presetChipSelected: {
    backgroundColor: Colors.primary,
  },
  presetChipText: {
    color: Colors.text,
    fontSize: FontSizes.sm,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  actionButton: {
    flex: 1,
    borderRadius: BorderRadius.md,
  },
  infoItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: Spacing.sm,
  },
  infoContent: {
    flex: 1,
    marginLeft: Spacing.md,
  },
  infoTitle: {
    fontSize: FontSizes.md,
    color: Colors.text,
    fontWeight: '600',
    marginBottom: 2,
  },
  infoText: {
    fontSize: FontSizes.sm,
    color: Colors.textSecondary,
    lineHeight: 18,
  },
  divider: {
    backgroundColor: Colors.border,
  },
  bottomSpacer: {
    height: Spacing.xl,
  },
});
