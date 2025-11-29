import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider as PaperProvider } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';

import OverviewScreen from './src/screens/OverviewScreen';
import TrainingScreen from './src/screens/TrainingScreen';
import GenerateScreen from './src/screens/GenerateScreen';
import PrivacyScreen from './src/screens/PrivacyScreen';
import AuditLogScreen from './src/screens/AuditLogScreen';
import SettingsScreen from './src/screens/SettingsScreen';

import { PaperTheme, Colors } from './src/theme';

const Tab = createBottomTabNavigator();

// Custom dark theme for navigation
const navigationTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    primary: Colors.primary,
    background: Colors.background,
    card: Colors.surface,
    text: Colors.text,
    border: Colors.border,
    notification: Colors.accent,
  },
};

const getTabIcon = (routeName, focused, color) => {
  let iconName;
  switch (routeName) {
    case 'Overview':
      iconName = focused ? 'view-dashboard' : 'view-dashboard-outline';
      break;
    case 'Train':
      iconName = focused ? 'brain' : 'brain';
      break;
    case 'Generate':
      iconName = focused ? 'creation' : 'creation-outline';
      break;
    case 'Privacy':
      iconName = focused ? 'shield-lock' : 'shield-lock-outline';
      break;
    case 'Audit':
      iconName = focused ? 'file-document' : 'file-document-outline';
      break;
    case 'Settings':
      iconName = focused ? 'cog' : 'cog-outline';
      break;
    default:
      iconName = 'circle';
  }
  return <MaterialCommunityIcons name={iconName} size={22} color={color} />;
};

export default function App() {
  return (
    <PaperProvider theme={PaperTheme}>
      <StatusBar style="light" />
      <NavigationContainer theme={navigationTheme}>
        <Tab.Navigator
          initialRouteName="Overview"
          screenOptions={({ route }) => ({
            headerShown: true,
            headerStyle: {
              backgroundColor: Colors.surface,
              elevation: 0,
              shadowOpacity: 0,
              borderBottomWidth: 1,
              borderBottomColor: Colors.border,
            },
            headerTintColor: Colors.text,
            headerTitleStyle: {
              fontWeight: '600',
            },
            tabBarStyle: {
              backgroundColor: Colors.surface,
              borderTopColor: Colors.border,
              borderTopWidth: 1,
              paddingTop: 8,
              paddingBottom: 8,
              height: 65,
            },
            tabBarActiveTintColor: Colors.primary,
            tabBarInactiveTintColor: Colors.textMuted,
            tabBarLabelStyle: {
              fontSize: 11,
              fontWeight: '500',
              marginTop: 4,
            },
            tabBarIcon: ({ focused, color }) => getTabIcon(route.name, focused, color),
          })}
        >
          <Tab.Screen 
            name="Overview" 
            component={OverviewScreen}
            options={{ title: 'Dashboard' }}
          />
          <Tab.Screen 
            name="Train" 
            component={TrainingScreen}
            options={{ title: 'Training' }}
          />
          <Tab.Screen 
            name="Generate" 
            component={GenerateScreen}
            options={{ title: 'Generate' }}
          />
          <Tab.Screen 
            name="Privacy" 
            component={PrivacyScreen}
            options={{ title: 'Privacy' }}
          />
          <Tab.Screen 
            name="Audit" 
            component={AuditLogScreen}
            options={{ title: 'Audit Log' }}
          />
          <Tab.Screen 
            name="Settings" 
            component={SettingsScreen}
            options={{ title: 'Settings' }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}
