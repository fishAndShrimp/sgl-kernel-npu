def qsa_store_cache_rows(cache, slot_mapping, rows):
    slots = slot_mapping.reshape(-1)[: rows.shape[0]].long()
    rows = rows.reshape(rows.shape[0], cache.shape[-1])[: slots.numel()]
    pages, page_size = cache.shape[:2]
    valid = (slots >= 0) & (slots < pages * page_size)
    slots = slots[valid]
    cache[slots // page_size, slots % page_size, 0] = rows[valid]
