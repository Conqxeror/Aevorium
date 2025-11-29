import React from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { Colors, Spacing, BorderRadius } from '../theme';

/**
 * Skeleton loader component for smooth loading states
 */
export function SkeletonLoader({ width = '100%', height = 20, style, borderRadius }) {
  const animatedValue = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(animatedValue, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(animatedValue, {
          toValue: 0,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    );
    animation.start();
    return () => animation.stop();
  }, [animatedValue]);

  const opacity = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: [0.3, 0.7],
  });

  return (
    <Animated.View
      style={[
        styles.skeleton,
        {
          width,
          height,
          borderRadius: borderRadius || BorderRadius.md,
          opacity,
        },
        style,
      ]}
    />
  );
}

/**
 * Skeleton for metric cards
 */
export function MetricCardSkeleton() {
  return (
    <View style={styles.metricCard}>
      <View style={styles.metricHeader}>
        <SkeletonLoader width={80} height={12} />
        <SkeletonLoader width={24} height={24} borderRadius={BorderRadius.sm} />
      </View>
      <SkeletonLoader width={60} height={28} style={{ marginTop: Spacing.sm }} />
      <SkeletonLoader width={100} height={10} style={{ marginTop: Spacing.xs }} />
    </View>
  );
}

/**
 * Skeleton for section cards
 */
export function SectionCardSkeleton({ lines = 3 }) {
  return (
    <View style={styles.sectionCard}>
      <View style={styles.sectionHeader}>
        <SkeletonLoader width={24} height={24} borderRadius={BorderRadius.sm} />
        <SkeletonLoader width={120} height={18} style={{ marginLeft: Spacing.sm }} />
      </View>
      <View style={styles.sectionContent}>
        {Array.from({ length: lines }).map((_, i) => (
          <SkeletonLoader 
            key={i} 
            width={`${100 - i * 15}%`} 
            height={14} 
            style={{ marginBottom: Spacing.sm }} 
          />
        ))}
      </View>
    </View>
  );
}

/**
 * Skeleton for list items (audit log entries)
 */
export function ListItemSkeleton() {
  return (
    <View style={styles.listItem}>
      <SkeletonLoader width={32} height={32} borderRadius={16} />
      <View style={styles.listContent}>
        <SkeletonLoader width={140} height={14} />
        <SkeletonLoader width={200} height={12} style={{ marginTop: Spacing.xs }} />
      </View>
      <SkeletonLoader width={60} height={10} />
    </View>
  );
}

/**
 * Dashboard loading skeleton
 */
export function DashboardSkeleton() {
  return (
    <View style={styles.dashboard}>
      {/* Header skeleton */}
      <SkeletonLoader 
        width="100%" 
        height={100} 
        borderRadius={0} 
        style={{ marginBottom: Spacing.md }} 
      />
      
      {/* Metrics row */}
      <View style={styles.metricsRow}>
        <MetricCardSkeleton />
        <MetricCardSkeleton />
      </View>
      <View style={styles.metricsRow}>
        <MetricCardSkeleton />
        <MetricCardSkeleton />
      </View>
      
      {/* Section */}
      <SectionCardSkeleton lines={4} />
    </View>
  );
}

const styles = StyleSheet.create({
  skeleton: {
    backgroundColor: Colors.surfaceLight,
  },
  metricCard: {
    backgroundColor: Colors.cardBackground,
    borderRadius: BorderRadius.lg,
    padding: Spacing.md,
    flex: 1,
    margin: Spacing.xs,
  },
  metricHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  sectionCard: {
    backgroundColor: Colors.cardBackground,
    borderRadius: BorderRadius.lg,
    marginVertical: Spacing.sm,
    overflow: 'hidden',
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  sectionContent: {
    padding: Spacing.md,
  },
  listItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  listContent: {
    flex: 1,
    marginLeft: Spacing.md,
  },
  dashboard: {
    padding: Spacing.md,
  },
  metricsRow: {
    flexDirection: 'row',
    marginHorizontal: -Spacing.xs,
    marginBottom: Spacing.sm,
  },
});
