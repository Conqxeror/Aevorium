import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { Card } from 'react-native-paper';
import api from '../components/ApiClient';

export default function AuditLogScreen(){
  const [logs, setLogs] = useState([]);
  useEffect(()=>{ loadLogs() },[]);
  const loadLogs = async () => {
    try {
      const r = await api.get('/audit-log');
      setLogs(r.data || []);
    } catch (e) {
      setLogs([]);
    }
  }

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.card}>
        <Card.Title title="Audit Log" />
        <Card.Content>
          {logs && logs.length>0 ? logs.slice(-50).reverse().map((log, idx) => (
            <View key={idx} style={{ marginVertical: 6 }}>
              <Text style={{fontWeight:'bold'}}>{log.timestamp} - {log.event_type}</Text>
              <Text>{JSON.stringify(log.details)}</Text>
            </View>
          )) : <Text>No log events</Text>}
        </Card.Content>
      </Card>
    </ScrollView>
  )
}

const styles = StyleSheet.create({ container: { flex:1, padding: 10 }, card: { marginVertical: 8 } });
