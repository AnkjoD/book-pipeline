-- This test returns rows only when the derived clean fields violate their contract.

select
    book_id,
    pages_clean,
    is_pages_suspicious,
    release_year,
    is_release_year_suspicious
from {{ ref('dim_books') }}
where (pages_clean is not null and (pages_clean <= 0 or pages_clean > 10000))
   or (is_pages_suspicious = false and pages is not null and (pages <= 0 or pages > 10000))
   or (
       is_release_year_suspicious = false
       and release_year is not null
       and (release_year < -5000 or release_year > year(current_date) + 1)
   )
