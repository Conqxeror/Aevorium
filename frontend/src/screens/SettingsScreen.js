import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, Alert, Linking } from 'react-native';
import { Text, TextInput, Button, Switch, Divider, List } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import GradientHeader from '../components/GradientHeader';
import SectionCard from '../components/SectionCard';
import { Colors, Spacing, FontSizes, BorderRadius } from '../theme';
import { getApiBaseUrl } from '../components/ApiClient';

const APP_VERSION = '1.0.0';
const BUILD_NUMBER = '1';

export default function SettingsScreen() {
  const [apiUrl, setApiUrl] = useState('');
  const [savedUrl, setSavedUrl] = useState('');
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState(null);
  
  // Settings
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const stored = await AsyncStorage.getItem('settings');
      if (stored) {
        const settings = JSON.parse(stored);
        setAutoRefresh(settings.autoRefresh ?? true);
        setNotifications(settings.notifications ?? true);
        setDarkMode(settings.darkMode ?? true);
      }
      
      const storedUrl = await AsyncStorage.getItem('apiUrl');
      if (storedUrl) {
        setApiUrl(storedUrl);
        setSavedUrl(storedUrl);
      } else {
        setApiUrl(getApiBaseUrl());
        setSavedUrl(getApiBaseUrl());
      }
    } catch (e) {
      console.log('Failed to load settings:', e);
    }
  };

  const saveSettings = async () => {
    try {
      await AsyncStorage.setItem('settings', JSON.stringify({
        autoRefresh,
        notifications,
        darkMode,
      }));
    } catch (e) {
      console.log('Failed to save settings:', e);
    }
  };

  useEffect(() => {
    saveSettings();
  }, [autoRefresh, notifications, darkMode]);

  const testConnection = async () => {
    if (!apiUrl.trim()) {
      Alert.alert('Error', 'Please enter an API URL');
      return;
    }

    setTestingConnection(true);
    setConnectionStatus(null);

    try {
      const response = await fetch(`${apiUrl.trim()}/health`, {
        method: 'GET',
        timeout: 5000,
      });

      if (response.ok) {
        setConnectionStatus({ success: true, message: 'Connection successful!' });
      } else {
        setConnectionStatus({ success: false, message: `Server returned ${response.status}` });
      }
    } catch (e) {
      setConnectionStatus({ success: false, message: e.message || 'Connection failed' });
    } finally {
      setTestingConnection(false);
    }
  };

  const saveApiUrl = async () => {
    try {
      await AsyncStorage.setItem('apiUrl', apiUrl.trim());
      setSavedUrl(apiUrl.trim());
      Alert.alert('Saved', 'API URL saved. Restart the app to apply changes.');
    } catch (e) {
      Alert.alert('Error', 'Failed to save API URL');
    }
  };

  const resetToDefault = () => {
    Alert.alert(
      'Reset Settings',
      'This will reset all settings to their default values.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: async () => {
            try {
              await AsyncStorage.clear();
              loadSettings();
              Alert.alert('Success', 'Settings reset to defaults');
            } catch (e) {
              Alert.alert('Error', 'Failed to reset settings');
            }
          },
        },
      ]
    );
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <GradientHeader
        title="Settings"
        subtitle="Configure app preferences"
        icon="cog"
      />

      {/* API Configuration */}
      <SectionCard title="API Configuration" icon="server">
        <Text style={styles.label}>Server URL</Text>
        <TextInput
          mode="outlined"
          value={apiUrl}
          onChangeText={setApiUrl}
          placeholder="http://192.168.1.100:8000"
          style={styles.input}
          outlineColor={Colors.border}
          activeOutlineColor={Colors.primary}
          textColor={Colors.text}
          autoCapitalize="none"
          autoCorrect={false}
          right={<TextInput.Icon icon="link" color={Colors.textMuted} />}
        />
        
        {connectionStatus && (
          <View style={[
            styles.statusBanner,
            { backgroundColor: connectionStatus.success ? Colors.success + '20' : Colors.danger + '20' }
          ]}>
            <MaterialCommunityIcons
              name={connectionStatus.success ? 'check-circle' : 'alert-circle'}
              size={20}
              color={connectionStatus.success ? Colors.success : Colors.danger}
            />
            <Text style={[
              styles.statusText,
              { color: connectionStatus.success ? Colors.success : Colors.danger }
            ]}>
              {connectionStatus.message}
            </Text>
          </View>
        )}

        <View style={styles.buttonRow}>
          <Button
            mode="outlined"
            onPress={testConnection}
            loading={testingConnection}
            disabled={testingConnection}
            style={styles.button}
            icon="connection"
            textColor={Colors.text}
          >
            Test
          </Button>
          <Button
            mode="contained"
            onPress={saveApiUrl}
            disabled={apiUrl === savedUrl || testingConnection}
            style={styles.button}
            icon="content-save"
            buttonColor={Colors.primary}
          >
            Save
          </Button>
        </View>

        <Text style={styles.hint}>
          💡 Enter your server's IP address. Make sure your phone is on the same network.
        </Text>
      </SectionCard>

      {/* App Preferences */}
      <SectionCard title="Preferences" icon="tune">
        <List.Item
          title="Auto-refresh data"
          description="Automatically refresh dashboard data"
          left={(props) => <List.Icon {...props} icon="refresh-auto" color={Colors.primary} />}
          right={() => (
            <Switch
              value={autoRefresh}
              onValueChange={setAutoRefresh}
              color={Colors.primary}
            />
          )}
          titleStyle={styles.listTitle}
          descriptionStyle={styles.listDescription}
        />
        <Divider style={styles.divider} />
        
        <List.Item
          title="Push notifications"
          description="Get notified on training completion"
          left={(props) => <List.Icon {...props} icon="bell" color={Colors.chartOrange} />}
          right={() => (
            <Switch
              value={notifications}
              onValueChange={setNotifications}
              color={Colors.primary}
            />
          )}
          titleStyle={styles.listTitle}
          descriptionStyle={styles.listDescription}
        />
        <Divider style={styles.divider} />
        
        <List.Item
          title="Dark mode"
          description="Use dark theme (recommended)"
          left={(props) => <List.Icon {...props} icon="theme-light-dark" color={Colors.chartPurple} />}
          right={() => (
            <Switch
              value={darkMode}
              onValueChange={setDarkMode}
              color={Colors.primary}
              disabled={true}
            />
          )}
          titleStyle={styles.listTitle}
          descriptionStyle={styles.listDescription}
        />
      </SectionCard>

      {/* About */}
      <SectionCard title="About Aevorium" icon="information">
        <View style={styles.aboutHeader}>
          <MaterialCommunityIcons name="brain" size={48} color={Colors.primary} />
          <View style={styles.aboutInfo}>
            <Text style={styles.appName}>Aevorium</Text>
            <Text style={styles.appTagline}>Federated Synthetic Data Platform</Text>
          </View>
        </View>

        <View style={styles.versionRow}>
          <Text style={styles.versionLabel}>Version</Text>
          <Text style={styles.versionValue}>{APP_VERSION} ({BUILD_NUMBER})</Text>
        </View>
        <Divider style={styles.divider} />

        <Text style={styles.description}>
          Aevorium enables collaborative machine learning on sensitive healthcare data 
          using federated learning and differential privacy. Generate high-fidelity 
          synthetic datasets without exposing raw patient records.
        </Text>

        <View style={styles.techStack}>
          <Text style={styles.techTitle}>Technology Stack</Text>
          <View style={styles.techRow}>
            <View style={styles.techItem}>
              <MaterialCommunityIcons name="flower" size={20} color={Colors.chartGreen} />
              <Text style={styles.techName}>Flower FL</Text>
            </View>
            <View style={styles.techItem}>
              <MaterialCommunityIcons name="shield-lock" size={20} color={Colors.chartPurple} />
              <Text style={styles.techName}>Opacus DP</Text>
            </View>
            <View style={styles.techItem}>
              <MaterialCommunityIcons name="fire" size={20} color={Colors.chartOrange} />
              <Text style={styles.techName}>PyTorch</Text>
            </View>
            <View style={styles.techItem}>
              <MaterialCommunityIcons name="api" size={20} color={Colors.chartBlue} />
              <Text style={styles.techName}>FastAPI</Text>
            </View>
          </View>
        </View>
      </SectionCard>

      {/* Actions */}
      <SectionCard title="Data & Privacy" icon="shield-check">
        <Button
          mode="outlined"
          onPress={resetToDefault}
          icon="restore"
          textColor={Colors.warning}
          style={styles.resetButton}
        >
          Reset All Settings
        </Button>
        <Text style={styles.resetHint}>
          This will clear saved preferences and API configuration
        </Text>
      </SectionCard>

      {/* Links */}
      <SectionCard title="Resources" icon="link-variant">
        <List.Item
          title="Documentation"
          description="Learn how to use Aevorium"
          left={(props) => <List.Icon {...props} icon="book-open-variant" color={Colors.chartBlue} />}
          right={(props) => <List.Icon {...props} icon="chevron-right" color={Colors.textMuted} />}
          onPress={() => Linking.openURL('https://github.com/Conqxeror/Aevorium')}
          titleStyle={styles.listTitle}
          descriptionStyle={styles.listDescription}
        />
        <Divider style={styles.divider} />
        
        <List.Item
          title="GitHub Repository"
          description="View source code and contribute"
          left={(props) => <List.Icon {...props} icon="github" color={Colors.text} />}
          right={(props) => <List.Icon {...props} icon="chevron-right" color={Colors.textMuted} />}
          onPress={() => Linking.openURL('https://github.com/Conqxeror/Aevorium')}
          titleStyle={styles.listTitle}
          descriptionStyle={styles.listDescription}
        />
        <Divider style={styles.divider} />
        
        <List.Item
          title="API Documentation"
          description="Swagger/OpenAPI reference"
          left={(props) => <List.Icon {...props} icon="api" color={Colors.chartGreen} />}
          right={(props) => <List.Icon {...props} icon="chevron-right" color={Colors.textMuted} />}
          onPress={() => Linking.openURL(`${apiUrl}/docs`)}
          titleStyle={styles.listTitle}
          descriptionStyle={styles.listDescription}
        />
      </SectionCard>

      <View style={styles.footer}>
        <Text style={styles.footerText}>Made with ❤️ for healthcare privacy</Text>
        <Text style={styles.copyright}>© 2025 Aevorium Project</Text>
      </View>

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
  statusBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.sm,
    borderRadius: BorderRadius.md,
    marginBottom: Spacing.sm,
  },
  statusText: {
    marginLeft: Spacing.sm,
    fontSize: FontSizes.sm,
    fontWeight: '500',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
    marginBottom: Spacing.sm,
  },
  button: {
    flex: 1,
    borderRadius: BorderRadius.md,
  },
  hint: {
    fontSize: FontSizes.xs,
    color: Colors.textMuted,
    marginTop: Spacing.xs,
  },
  listTitle: {
    color: Colors.text,
    fontSize: FontSizes.md,
  },
  listDescription: {
    color: Colors.textMuted,
    fontSize: FontSizes.sm,
  },
  divider: {
    backgroundColor: Colors.border,
  },
  aboutHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  aboutInfo: {
    marginLeft: Spacing.md,
  },
  appName: {
    fontSize: FontSizes.xl,
    fontWeight: 'bold',
    color: Colors.text,
  },
  appTagline: {
    fontSize: FontSizes.sm,
    color: Colors.textSecondary,
  },
  versionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: Spacing.sm,
  },
  versionLabel: {
    color: Colors.textSecondary,
    fontSize: FontSizes.md,
  },
  versionValue: {
    color: Colors.text,
    fontSize: FontSizes.md,
    fontWeight: '500',
  },
  description: {
    fontSize: FontSizes.sm,
    color: Colors.textSecondary,
    lineHeight: 20,
    marginTop: Spacing.md,
  },
  techStack: {
    marginTop: Spacing.lg,
  },
  techTitle: {
    fontSize: FontSizes.sm,
    color: Colors.textMuted,
    fontWeight: '600',
    marginBottom: Spacing.sm,
  },
  techRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.md,
  },
  techItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surfaceLight,
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.md,
  },
  techName: {
    fontSize: FontSizes.xs,
    color: Colors.textSecondary,
    marginLeft: Spacing.xs,
  },
  resetButton: {
    borderColor: Colors.warning,
    borderRadius: BorderRadius.md,
  },
  resetHint: {
    fontSize: FontSizes.xs,
    color: Colors.textMuted,
    marginTop: Spacing.sm,
    textAlign: 'center',
  },
  footer: {
    alignItems: 'center',
    marginTop: Spacing.lg,
    paddingVertical: Spacing.md,
  },
  footerText: {
    fontSize: FontSizes.sm,
    color: Colors.textSecondary,
  },
  copyright: {
    fontSize: FontSizes.xs,
    color: Colors.textMuted,
    marginTop: Spacing.xs,
  },
  bottomSpacer: {
    height: Spacing.xl,
  },
});
