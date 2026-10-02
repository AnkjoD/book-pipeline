-- Current-state book metrics fact.
-- Grain: one row per book_id. This model does not retain metric history.

select
    id as book_id,
    release_year,
    case
        when release_year is not null
         and (
             release_year < -5000
             or release_year > year(current_date) + 1
         ) then true
        else false
    end as is_release_year_suspicious,
    literary_type,
    book_category,
    pages,
    case
        when pages between 1 and 10000 then pages
        else null
    end as pages_clean,
    case
        when pages is not null and (pages <= 0 or pages > 10000) then true
        else false
    end as is_pages_suspicious,
    rating,
    ratings_count,
    reviews_count,
    users_count,
    users_read_count,
    editions_count,
    activities_count,
    updated_at,
    current_timestamp as warehouse_loaded_at
from {{ source('silver', 'silver_books') }}
where id is not null
