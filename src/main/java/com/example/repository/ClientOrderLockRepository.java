package com.example.repository;

import com.example.entity.ClientOrderLock;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface ClientOrderLockRepository extends JpaRepository<ClientOrderLock, Long> {

    @Query("SELECT col FROM ClientOrderLock col WHERE col.orderId = :orderId AND col.isActive = true AND col.expiresAt > :now")
    Optional<ClientOrderLock> findActiveLockByOrderId(@Param("orderId") Long orderId, @Param("now") LocalDateTime now);

    @Query("SELECT col FROM ClientOrderLock col WHERE col.lockToken = :lockToken AND col.isActive = true")
    Optional<ClientOrderLock> findActiveLockByToken(@Param("lockToken") String lockToken);

    @Query("SELECT col FROM ClientOrderLock col WHERE col.lockedBy = :lockedBy AND col.isActive = true")
    List<ClientOrderLock> findActiveLocksByUser(@Param("lockedBy") String lockedBy);

    @Modifying
    @Query("UPDATE ClientOrderLock col SET col.isActive = false WHERE col.orderId = :orderId AND col.isActive = true")
    int releaseAllLocksForOrder(@Param("orderId") Long orderId);

    @Modifying
    @Query("UPDATE ClientOrderLock col SET col.isActive = false WHERE col.lockToken = :lockToken AND col.isActive = true")
    int releaseLockByToken(@Param("lockToken") String lockToken);

    @Modifying
    @Query("UPDATE ClientOrderLock col SET col.isActive = false WHERE col.expiresAt < :now AND col.isActive = true")
    int cleanupExpiredLocks(@Param("now") LocalDateTime now);

    @Query("SELECT COUNT(col) FROM ClientOrderLock col WHERE col.orderId = :orderId AND col.isActive = true AND col.expiresAt > :now")
    int countActiveLocksForOrder(@Param("orderId") Long orderId, @Param("now") LocalDateTime now);

    @Query("SELECT col FROM ClientOrderLock col WHERE col.isActive = true AND col.expiresAt < :now")
    List<ClientOrderLock> findAllExpiredLocks(@Param("now") LocalDateTime now);
} 