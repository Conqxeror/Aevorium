import React from 'react';
import { View, StyleSheet, TouchableOpacity } from 'react-native';
import { Text } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Colors, Spacing, FontSizes, BorderRadius } from '../theme';

/**
 * ConnectionBanner - Shows persistent API connection status
 */
export default function ConnectionBanner({ 
  isConnected, 
  message = '', 
  onRetry,
  latency 
}) {
  if (isConnected) {
    return null; // Don't show banner when connected
  }

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <MaterialCommunityIcons 
          name="cloud-off-outline" 
          size={18} 
          color={Colors.danger} 
        />
        <Text style={styles.message}>
          {message || 'Unable to connect to server'}
        </Text>
      </View>
      {onRetry && (
        <TouchableOpacity style={styles.retryButton} onPress={onRetry}>
          <MaterialCommunityIcons name="refresh" size={16} color={Colors.text} />
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

/**
 * Mini connection indicator for headers
 */
export function ConnectionDot({ isConnected }) {
  return (
    <View style={styles.dotContainer}>
      <View style={[
        styles.dot,
        { backgroundColor: isConnected ? Colors.success : Colors.danger }
      ]} />
      <Text style={[
        styles.dotText,
        { color: isConnected ? Colors.success : Colors.danger }
      ]}>
        {isConnected ? 'Connected' : 'Offline'}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.danger + '20',
    borderBottomWidth: 1,
    borderBottomColor: Colors.danger + '40',
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  message: {
    color: Colors.danger,
    fontSize: FontSizes.sm,
    marginLeft: Spacing.sm,
    flex: 1,
  },
  retryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.danger + '30',
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.md,
  },
  retryText: {
    color: Colors.text,
    fontSize: FontSizes.xs,
    marginLeft: Spacing.xs,
    fontWeight: '500',
  },
  dotContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: Spacing.xs,
  },
  dotText: {
    fontSize: FontSizes.xs,
    fontWeight: '500',
  },
});
