import { openDB, type IDBPDatabase } from 'idb';

const DB_NAME = 'layers-insight';
const STORE_NAME = 'metrics';
const DB_VERSION = 1;

let db: IDBPDatabase | null = null;

async function getDB(): Promise<IDBPDatabase> {
  if (!db) {
    db = await openDB(DB_NAME, DB_VERSION, {
      upgrade(database) {
        if (!database.objectStoreNames.contains(STORE_NAME)) {
          database.createObjectStore(STORE_NAME);
        }
      },
    });
  }
  return db;
}

export async function cacheMetrics(taskId: string, data: any): Promise<void> {
  try {
    const database = await getDB();
    await database.put(STORE_NAME, data, taskId);
  } catch (e) {
    console.warn('Failed to cache metrics:', e);
  }
}

export async function getCachedMetrics(taskId: string): Promise<any | null> {
  try {
    const database = await getDB();
    return await database.get(STORE_NAME, taskId) ?? null;
  } catch (e) {
    console.warn('Failed to get cached metrics:', e);
    return null;
  }
}

export async function clearMetricsCache(): Promise<void> {
  try {
    const database = await getDB();
    await database.clear(STORE_NAME);
  } catch (e) {
    console.warn('Failed to clear metrics cache:', e);
  }
}
