-- Tag dimension. Grain: one row per tag_id.

select
    id as tag_id,
    coalesce(nullif(trim(tag), ''), '[Unknown tag]') as tag_name,
    slug as tag_slug,
    tag_category_id,
    coalesce(nullif(trim(tag_category), ''), 'Unknown') as tag_category,
    tag_category_slug,
    "count" as global_count,
    case
        when tag is null or trim(tag) = '' then true
        else false
    end as is_tag_name_missing
from {{ source('silver', 'silver_tags') }}
where id is not null
