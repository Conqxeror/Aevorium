import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Colors, Spacing, FontSizes, BorderRadius } from '../theme';

/**
 * DataQualityCard - Shows synthetic data quality score
 */
export default function DataQualityCard({ 
  overallScore,
  continuousScore,
  categoricalScore,
  loading = false 
}) {
  const getScoreColor = (score) => {
    if (score >= 80) return Colors.success;
    if (score >= 60) return Colors.warning;
    return Colors.danger;
  };

  const getScoreLabel = (score) => {
    if (score >= 90) return 'Excellent';
    if (score >= 80) return 'Good';
    if (score >= 60) return 'Fair';
    return 'Needs Improvement';
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>Validating data quality...</Text>
      </View>
    );
  }

  if (overallScore === null || overallScore === undefined) {
    return null;
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <MaterialCommunityIcons 
          name="check-decagram" 
          size={24} 
          color={getScoreColor(overallScore)} 
        />
        <Text style={styles.title}>Data Quality Score</Text>
      </View>

      {/* Overall Score */}
      <View style={styles.mainScore}>
        <Text style={[styles.scoreValue, { color: getScoreColor(overallScore) }]}>
          {overallScore.toFixed(1)}%
        </Text>
        <Text style={[styles.scoreLabel, { color: getScoreColor(overallScore) }]}>
          {getScoreLabel(overallScore)}
        </Text>
      </View>

      {/* Score Breakdown */}
      <View style={styles.breakdown}>
        <View style={styles.breakdownItem}>
          <View style={styles.breakdownHeader}>
            <MaterialCommunityIcons name="chart-line" size={16} color={Colors.chartBlue} />
            <Text style={styles.breakdownLabel}>Continuous</Text>
          </View>
          <View style={styles.progressContainer}>
            <View 
              style={[
                styles.progressBar, 
                { 
                  width: `${continuousScore || 0}%`,
                  backgroundColor: getScoreColor(continuousScore || 0) 
                }
              ]} 
            />
          </View>
          <Text style={styles.breakdownValue}>{(continuousScore || 0).toFixed(1)}%</Text>
        </View>

        <View style={styles.breakdownItem}>
          <View style={styles.breakdownHeader}>
            <MaterialCommunityIcons name="shape" size={16} color={Colors.chartPurple} />
            <Text style={styles.breakdownLabel}>Categorical</Text>
          </View>
          <View style={styles.progressContainer}>
            <View 
              style={[
                styles.progressBar, 
                { 
                  width: `${categoricalScore || 0}%`,
                  backgroundColor: getScoreColor(categoricalScore || 0) 
                }
              ]} 
            />
          </View>
          <Text style={styles.breakdownValue}>{(categoricalScore || 0).toFixed(1)}%</Text>
        </View>
      </View>

      <Text style={styles.hint}>
        Score measures statistical similarity between real and synthetic data distributions
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.cardBackground,
    borderRadius: BorderRadius.lg,
    padding: Spacing.md,
    marginVertical: Spacing.sm,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  title: {
    fontSize: FontSizes.lg,
    fontWeight: '600',
    color: Colors.text,
    marginLeft: Spacing.sm,
  },
  mainScore: {
    alignItems: 'center',
    marginBottom: Spacing.lg,
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: 'bold',
  },
  scoreLabel: {
    fontSize: FontSizes.md,
    fontWeight: '500',
    marginTop: Spacing.xs,
  },
  breakdown: {
    gap: Spacing.md,
  },
  breakdownItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  breakdownHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    width: 100,
  },
  breakdownLabel: {
    fontSize: FontSizes.sm,
    color: Colors.textSecondary,
    marginLeft: Spacing.xs,
  },
  progressContainer: {
    flex: 1,
    height: 8,
    backgroundColor: Colors.surfaceLight,
    borderRadius: 4,
    marginHorizontal: Spacing.sm,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    borderRadius: 4,
  },
  breakdownValue: {
    fontSize: FontSizes.sm,
    color: Colors.text,
    fontWeight: '500',
    width: 50,
    textAlign: 'right',
  },
  hint: {
    fontSize: FontSizes.xs,
    color: Colors.textMuted,
    marginTop: Spacing.md,
    textAlign: 'center',
  },
  loadingText: {
    fontSize: FontSizes.sm,
    color: Colors.textSecondary,
    textAlign: 'center',
    padding: Spacing.md,
  },
});
