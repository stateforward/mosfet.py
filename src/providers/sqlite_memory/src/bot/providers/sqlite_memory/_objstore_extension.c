#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1

#include "objstore/objstore.h"

int sqlite3_botobjstore_init(sqlite3 *db, char **error, const sqlite3_api_routines *api) {
  SQLITE_EXTENSION_INIT2(api);
  (void)error;
  objstore_config config = {
      .backend = OBJSTORE_BACKEND_SQLITE,
      .storage_root = NULL,
      .chunk_size_bytes = 0,
      .shard_width = 0,
      .sync_mode = OBJSTORE_SYNC_FULL,
      .reserved_flags = 0,
  };
  return objstore_register(db, &config);
}

int sqlite3_extension_init(sqlite3 *db, char **error, const sqlite3_api_routines *api) {
  return sqlite3_botobjstore_init(db, error, api);
}
